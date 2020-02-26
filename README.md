# SimCLR

Pytorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709) on CIFAR-10.

## Installation
Runs on Python 3 and depends on the following Python packages
* [Pytorch](https://pytorch.org)
* [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
* [Pytorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Usage

To train run `python train.py`. Optional flags are as follows:
* `--TEMP`: Temperature parameter of NT-Xent
* `--DISTORT_STRENGTH`: Strength of the colour distortion
* `--LOG_INT`: Controls when updates are displayed
* `--SAVE_NAME`: Saved model name, stored in the `/ckpt` folder. 
* `--EPOCHS`
* `--BATCH_SIZE`

To test a linear classifier on top of the trained model run `python test.py`. Optional flags are as follows:
* `--SAVED_MODEL`: Path of saved model
* `--LOG_INT`
* `--EPOCHS`
* `--BATCH_SIZE`