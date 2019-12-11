# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
import numpy as np
import pdb
import argparse
import time
import os
import sys
import foolbox
import wideresnet
from collections import OrderedDict
from utils import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def remove_module_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict

# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"])

# EBM specific
parser.add_argument("--n_steps", type=int, default=100)
parser.add_argument("--width", type=int, default=10)
parser.add_argument("--depth", type=int, default=28)
# 
parser.add_argument('--n_steps_refine', type=int, default=0)
parser.add_argument('--n_classes',type=int,default=10)
parser.add_argument('--init_batch_size', type=int, default=128)
parser.add_argument('--softmax_ce', action='store_true')
# attack
parser.add_argument('--attack_conf',  action='store_true')
parser.add_argument('--random_init',  action='store_true')
parser.add_argument('--threshold', type=float, default=.7)
parser.add_argument('--debug',  action='store_true')
parser.add_argument('--no_random_start',  action='store_true')
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--distance", type=str, default='Linf')
parser.add_argument("--n_steps_pgd_attack", type=int, default=40)
parser.add_argument("--start_batch", type=int, default=-1)
parser.add_argument("--end_batch", type=int, default=2)
parser.add_argument("--sgld_sigma", type=float, default=1e-2)
parser.add_argument("--n_dup_chains", type=int, default=5)
parser.add_argument("--sigma", type=float, default=.03)
parser.add_argument("--base_dir", type=str, default='./adv_results')

# logging

parser.add_argument('--exp_name', type=str, default='exp', help='saves everything in ?r/exp_name/')
args = parser.parse_args()
device = torch.device('cuda')
args_ = vars(args)
for key in args_.keys():
    print('{}:   {}'.format(key,args_[key]))


base_dir = args.base_dir

save_dir = os.path.join(base_dir, args.exp_name, 'saved_model')
last_dir = os.path.join(save_dir,'last')
best_dir = os.path.join(save_dir,'best')
data_dir = os.path.join(base_dir,'data')


class gradient_attack_wrapper(nn.Module):
  def __init__(self, model):
    super(gradient_attack_wrapper, self).__init__()
    self.model = model.eval()

  def forward(self, x):
    x = x - 0.5
    x = x / 0.5
    x.requires_grad_()
    out = self.model.module.refined_logits(x)
    return out

  def eval(self):
    return self.model.eval()

model_attack_wrapper =gradient_attack_wrapper

transformer_train  = transforms.Compose([transforms.ToTensor()])
transformer_test  = transforms.Compose([transforms.ToTensor()])

data_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=False,
                                                            transform=transformer_test, download=True),
                                           batch_size=args.batch_size, shuffle=False, num_workers=10)
init_loader = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=True,
                                                           download=True, transform=transformer_train),
                                          batch_size=args.init_batch_size, shuffle=True, num_workers=1)


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, 10)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z)


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None):
        super(CCF, self).__init__(depth, width, norm=norm)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, y[:, None])

# construct model and ship to GPU
f = CCF(args.depth, args.width, args.norm)
print(args.load_path)
print("loading model from {args.load_path}")
ckpt_dict = torch.load(args.load_path)
if "model_state_dict" in ckpt_dict:
    # loading from a new checkpoint
    f.load_state_dict(ckpt_dict["model_state_dict"])
else:
    # loading from an old checkpoint
    f.load_state_dict(ckpt_dict)

# wrapper class to provide utilities for what you need
class DummyModel(nn.Module):
    def __init__(self, f):
        super(DummyModel, self).__init__()
        self.f = f

    def logits(self, x):
        return self.f.classify(x)

    def refined_logits(self, x, n_steps=args.n_steps_refine):
        xs = x.size()
        dup_x = x.view(xs[0], 1, xs[1], xs[2], xs[3]).repeat(1, args.n_dup_chains, 1, 1, 1)
        dup_x = dup_x.view(xs[0] * args.n_dup_chains, xs[1], xs[2], xs[3])
        dup_x = dup_x + torch.randn_like(dup_x) * args.sigma
        refined = self.refine(dup_x, n_steps=n_steps, detach=False)
        logits = self.logits(refined)
        logits = logits.view(x.size(0), args.n_dup_chains, logits.size(1))
        logits = logits.mean(1)
        return logits

    def classify(self, x):
        logits = self.logits(x)
        pred = logits.max(1)[1]
        return pred

    def logpx_score(self, x):
        # unnormalized logprob, unconditional on class
        return self.f(x)

    def refine(self, x, n_steps=args.n_steps_refine, detach=True):
        # runs a markov chain seeded at x, use n_steps=10
        x_k = torch.autograd.Variable(x, requires_grad=True) if detach else x
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + args.sgld_sigma * torch.randn_like(x_k)
        final_samples = x_k.detach() if detach else x_k
        return final_samples

    def grad_norm(self, x):
        x_k = torch.autograd.Variable(x, requires_grad=True)
        f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    def logpx_delta_score(self, x, n_steps=args.n_steps_refine):
        # difference in logprobs from input x and samples from a markov chain seeded at x
        #
        init_scores = self.f(x)
        x_r = self.refine(x, n_steps=n_steps)
        final_scores = self.f(x_r)
        # for real data final_score is only slightly higher than init_score
        return init_scores - final_scores

    def logp_grad_score(self, x):
        return -self.grad_norm(x)


f = DummyModel(f)
model = f.to(device)
model = nn.DataParallel(model).to(device)

model.eval()
## Define criterion
criterion = foolbox.criteria.Misclassification()

## Initiate attack and wrap model
model_wrapped = model_attack_wrapper(model)
fmodel = foolbox.models.PyTorchModel(model_wrapped, bounds=(0.,1.), num_classes=10, device=device)

if args.distance == 'L2':
    distance = foolbox.distances.MeanSquaredDistance
    attack = foolbox.attacks.L2BasicIterativeAttack(model=fmodel, criterion=criterion, distance=distance)
else:
    distance = foolbox.distances.Linfinity
    attack = foolbox.attacks.RandomStartProjectedGradientDescentAttack(model=fmodel, criterion=criterion, distance=distance)

print('Starting...')
for i, (img, label) in enumerate(data_loader):
    adversaries = []
    if i < args.start_batch:
        continue
    if i >= args.end_batch:
      break
    img = img.data.cpu().numpy()
    logits = model_wrapped(torch.from_numpy(img[:, :, :, :]).to(device))
    _, top = torch.topk(logits,k=2,dim=1)
    top = top.data.cpu().numpy()
    pred = top[:,0]
    for k in range(len(label)):
      im = img[k,:,:,:]
      orig_label = label[k].data.cpu().numpy()
      if pred[k] != orig_label:
        continue
      best_adv = None
      for ii in range(20):
          try:
            adversarial = attack(im, label=orig_label, unpack=False, random_start=True, iterations=args.n_steps_pgd_attack) 
            if ii == 0 or best_adv.distance > adversarial.distance:
                best_adv = adversarial
          except:
            continue
      try:
          adversaries.append((im, orig_label, adversarial.image, adversarial.adversarial_class))
      except:
          continue
    adv_save_dir = os.path.join(base_dir, args.exp_name)
    save_file = 'adversarials_batch_'+str(i)
    if not os.path.exists(os.path.join(adv_save_dir,save_file)):
        os.makedirs(os.path.join(adv_save_dir,save_file))
    np.save(os.path.join(adv_save_dir,save_file),adversaries)
