<div align="center">
<h1> Meta-BCR (Title) </h1>

<a href="https://jianqingzheng.github.io/meta_bcr/"><img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fjianqingzheng.github.io%2Fmeta_bcr%2F&up_message=online&up_color=darkcyan&down_message=offline&down_color=darkgray&label=Project%20Page"></a>
[![arXiv](https://img.shields.io/badge/arXiv-xxx-b31b1b.svg)]()
[![Explore MetaBCR in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/meta_bcr/blob/main/meta_bcr.ipynb)

</div>

<table>
  <tr>
    <td><img src="docs/static/images/demo_1.gif" alt="demo_fig1" width="100%" /></td>
    <td><img src="docs/static/images/demo_2.gif" alt="demo_fig2" width="100%" /></td>
  </tr>
</table>

Code for paper [Meta-BCR (Title)]()


> This repo provides an implementation of the training and inference pipeline of **Meta-BCR** based on Pytorch. 

---
### Contents ###
- [0. Brief Introduction](#0-brief-intro)
- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [2.1. Setup](#21-setup)
  - [2.2. Training](#22-training)
  - [2.3. Inference](#23-inference)
- [3. Demo](#3-demo)
- [4. Citing this work](#4-citing-this-work)


---

## 0. Brief Intro ##

![header](docs/static/images/graphic_abstract.png)
Abstract...


Highlight:
<ul style="width: auto; height: 200px; overflow: auto; padding:0.4em; margin:0em; text-align:justify; font-size:small">
  <li> <b>Point1</b>: xxx;
  </li>
  <li> <b>Point2</b>: xxx.
  </li>
</ul>

---
## 1. Installation ##

Clone code from Github repo: https://github.com/jianqingzheng/meta_bcr.git
```shell
git clone https://github.com/jianqingzheng/meta_bcr.git
cd meta_bcr/
```


install packages

[![OS](https://img.shields.io/badge/OS-Windows%7CLinux-darkblue)]()
[![PyPI pyversions](https://img.shields.io/badge/Python-3.8-blue)](https://pypi.python.org/pypi/ansicolortags/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1+cu113-lightblue)](https://pytorch.org/)
[![Numpy](https://img.shields.io/badge/Numpy-1.19.5-lightblue)](https://numpy.org)

```shell

```

> Other versions of the packages could also be applicable



---
## 2. Usage ##

### 2.1. Setup ###

Directory layout:
```
[$DOWNLOAD_DIR]/meta_bcr/ 
├── Config/
|   |   # configure file (.yaml files)
|   └── config_[$data_name].yaml
├── Data/
|   ├── /
|   └── ...
├── models/
|   └── ...
└── ...
```

Configuration setting:

<div align="center">
	
| Argument              | Example           | Description                                	|
| --------------------- | ----------------- |----------------------------------------------|
| `--data_name` 	    |'flu', 'rbd'        | The data  name                    |
| `--net_name` 	        |'acnn'             | The network name                    |
| `--ndims` 	        |2, 3                | The dimension of image                    |
</div>

> configuration settings are edited in `[$DOWNLOAD_DIR]/meta_bcr/Config/*.yaml`


### 2.2. Training ###

---
## 3. Demo ##

A demo can be found in the provided [notebook](https://github.com/jianqingzheng/meta_bcr/blob/main/meta_bcr.ipynb).

Alternatively, it can be easily run via [![Explore Meta-BCR in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/meta_bcr/blob/main/meta_bcr.ipynb).


---

## 4. Citing this work

Any publication that discloses findings arising from using this source code or the network model should cite:

```bibtex

```
