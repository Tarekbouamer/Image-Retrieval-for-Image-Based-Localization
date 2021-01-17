## Image-Retrieval-for-Image-Based-Localization

### TO DO?
* Add results on roxford5k rparis6k.

### Prerequisites
Main system requirements:
* Python3 (Tested with Python 3.6.9)
* Linux with GCC 7 or 8
* PyTorch 1.4.0 Torchvision 0.4.0
* CUDA (10.0 - 10.1)

### Setup

To install all other dependencies using pip:
```bash
pip install -r requirements.txt
```

Our code is split into two main components: a library containing implementations for the various network modules,
algorithms and utilities, and a set of scripts for training and testing the networks.

The library, called `cirtorch`, can be installed with:
```bash
git clone https://github.com/Tarekbouamer/Image-Retrieval-for-Image-Based-Localization.git
cd Image-Retrieval-for-Image-Based-Localization
python setup.py install
```
