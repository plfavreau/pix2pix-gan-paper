# pix2pix-gan

Coding this paper just for the fun of it, and because GANs are cool

Link to the paper -> https://arxiv.org/abs/1611.07004

## Getting started

### Download the dataset

Download & extract the dataset : http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz

!wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
!tar -xvf facades.tar.gz

Your workspace should look like this :
G:.
└───pix2pix-gan
└───facades
├───test
├───train
└───val

### Install the requirements

Install the requirements :

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```
