# JEM - Joint Energy Models

Code for the paper [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/abs/1912.03263).

We show that architectures traditionally used for classification can be reinterpreted as energy based models of the joint distribution over data and labels p(x, y). 

Training models in this way has many benifits including:

### Improved calibration
![calibration plots](figs/calib.png)

### Out-Of-Distribution Detection
![calibration plots](figs/ood.png)

### Adversarial Robustness
![calibration plots](figs/distal.png)

## Usage
### Training
To train a model on CIFAR10 as in the paper
```markdown
python train_wrn_ebm.py --lr .0001 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --width 10 --depth 28 --save_dir /YOUR/SAVE/DIR --plot_uncond --warmup_iters 1000
```




### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/wgrathwohl/JEM/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
