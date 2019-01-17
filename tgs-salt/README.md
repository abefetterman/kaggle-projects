# TGS Salt Identification Challenge

[Link to competition](https://www.kaggle.com/c/tgs-salt-identification-challenge)

Segment 2D earth ultrasound data to identify salt deposits.

## Model and Training

The model used is a SE-UNet with residual blocks. See [unetres.py](./unetres.py) for 
implementation details.

Data augmentation used from [albumentations](https://github.com/albu/albumentations): 
RandomSizedCrop, HorizontalFlip, GridDistortion, RandomContrast, 
RandomBrightness, RandomGamma

Supervised training was done in 3 phases: 
* Initial tuning, pixel-wise BCE loss function
* Fine tuning, Lovasz loss
* Second fine tuning, Lovasz loss with cosine annealing LR

Semi-supervised training was tried as well:
* Variational Adversarial Training plus Entropy Minimization

Additional tricks include using middle data to predict whether the image has no salt at all,
use of hypercolumns and deep supervised training.

## References

_U-Net: Convolutional Networks for Biomedical Image Segmentation._
Olaf Ronneberger, Philipp Fischer, Thomas Brox.
[arXiv:1505.04597](https://arxiv.org/abs/1505.04597) (2015)

_The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks_
M Berman, AR Triki, MB Blaschko. 
[arXiv:1705.08790](https://arxiv.org/abs/1705.08790) (2017)

_Squeeze-and-Excitation Networks._
J Hu, L Shen, S Albanie, et. al.
[arXiv:1709.01507](https://arxiv.org/abs/1709.01507) (2017)

_Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning._
T Miyato, S Maeda, M Koyama, S Ishii. 
[arXiv:1704.03976](https://arxiv.org/abs/1704.03976) (2017)