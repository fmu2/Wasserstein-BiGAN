# Wasserstein BiGAN

PyTorch implementation of bidirectional generative adversarial network ([BiGAN](https://arxiv.org/abs/1605.09782), a.k.a. [ALI](https://arxiv.org/abs/1606.00704)) trained using Wasserstein distance (see [WGAN](https://arxiv.org/abs/1701.07875) and [WGAN-GP](https://arxiv.org/abs/1704.00028)). The code has been tested in an conda environment with Python 3 and PyTorch >= 1.0.

## Overview
This repository contains code for training BiGAN on SVHN, CIFAR-10 and Celeba datasets. Our implementation is different from the original BiGAN/ALI implementation in the following ways:

* We normalize pixel values to [-1, 1].
* Our training objective is Wasserstein distance, not Jenson-Shannon divergence. Our model hence has a critic network instead of a discriminator network.
* The critic network does **NOT** use normalization layers (batch norm, instance norm, etc.). We found that training fails if we incorporate normalization into the critic network.
* We apply gradient penalty to stablize training.

## Quick Start
* Update the loading and saving paths.
* Check (and update) the hyperparameters.
* Train on SVHN
```shell
python ./wali_svhn.py
```
* Train on CIFAR-10
```shell
python ./wali_cifar10.py
```
* Train on Celeba
```shell
python .wali_celeba.py
```

## Results
All models are trained using default hyperparameter settings for 20,000 iterations.

Generation            |  Reconstruction
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/fmu2/Wasserstein-BiGAN/master/results/svhn_gen.png)  |  ![](https://raw.githubusercontent.com/fmu2/Wasserstein-BiGAN/master/results/svhn_rec.png)

Generation            |  Reconstruction
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/fmu2/Wasserstein-BiGAN/master/results/cifar10_gen.png)  |  ![](https://raw.githubusercontent.com/fmu2/Wasserstein-BiGAN/master/results/cifar10_rec.png)

Generation            |  Reconstruction
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/fmu2/Wasserstein-BiGAN/master/results/celeba_gen.png)  |  ![](https://raw.githubusercontent.com/fmu2/Wasserstein-BiGAN/master/results/celeba_rec.png)

## Contact
[Fangzhou Mu](http://pages.cs.wisc.edu/~fmu/) (fmu2@wisc.edu)

## Related Code Repositories
* BiGAN <https://github.com/jeffdonahue/bigan>
* ALI <https://github.com/IshmaelBelghazi/ALI>
* WGAN <https://github.com/martinarjovsky/WassersteinGAN>
* WGAN-GP <https://github.com/igul222/improved_wgan_training>

## References
```
@inproceedings{donahue2016adversarial,
  title={Adversarial feature learning},
  author={Donahue, Jeff and Kr{\"a}henb{\"u}hl, Philipp and Darrell, Trevor},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}

@inproceedings{dumoulin2016adversarially,
  title={Adversarially learned inference},
  author={Dumoulin, Vincent and Belghazi, Ishmael and Poole, Ben and Mastropietro, Olivier and Lamb, Alex and Arjovsky, Martin and Courville, Aaron},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}

@inproceedings{arjovsky2017wasserstein,
  title={Wasserstein gan},
  author={Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@inproceedings{gulrajani2017improved,
  title={Improved training of wasserstein gans},
  author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron C},
  booktitle={Advances in neural information processing systems (NeurIPS)},
  year={2017}
}
```
