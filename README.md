
#  Improving Self-Supervised Learning with Hardness-aware Dynamic Curriculum Learning: An Application to Digital Pathology
#### by [Chetan L. Srinidhi](https://srinidhipy.github.io) and [Anne L. Martel](https://medbio.utoronto.ca/faculty/martel)

* Official repository for [Improving Self-supervised Learning with Hardness-aware Dynamic Curriculum Learning: An Application to Digital Pathology](https://arxiv.org/abs/2108.07183). Accepted in **ICCV, 2021, CDpath workshop**, October, 2021. [[Conference proceedings]]() [[arXiv preprint]](https://arxiv.org/abs/2108.07183)

* <a href="https://github.com/srinidhiPY/ICCV-CDPATH2021-ID-8/tree/main/models"><img src="https://img.shields.io/badge/PRETRAINED-MODELS-<GREEN>.svg"/></a>

## Overview
In this work, we attempt to **improve self-supervised pretrained representations** through the lens of **curriculum learning** by proposing a **hardness-aware dynamic curriculum learning (HaDCL)** approach. To improve the robustness and generalizability of SSL, we dynamically leverage progressive harder examples via **easy-to-hard** and **hard-to-very-hard samples** during mini-batch downstream fine-tuning. We discover that by progressive stage-wise curriculum learning, the pretrained representations are significantly enhanced and adaptable to both **in-domain and out-of-domain** distribution data.

We carry out extensive validation experiments on **three** histopathology benchmark datasets on both **patch-wise** and **slide-level** classification tasks: 
- [Camelyon 16](https://camelyon16.grand-challenge.org)
- [Memorial Sloan Kettering Cancer Center (MSKCC), Breast Metastases dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52763339)
- [Colorectal Polyps classification](https://bmirds.github.io/MHIST/) 

## Method
<img src="Hadcl_figure.png" width="600px"/>

## Results
* Predicted tumor probability heat-maps on **Camelyon16 (in-domain)** and **MSKCC (out-of-domain)** test sets
<img src="Results.png" width="800px"/>

## Pre-requisites
Core implementation:
* Python 3.7+
* Pytorch 1.7+
* Openslide-python 1.1+
* Albumentations 1.8+
* Scikit-image 0.15+
* Scikit-learn 0.22+
* Matplotlib 3.2+
* Scipy, Numpy (any version)

