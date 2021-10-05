
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

## Datasets
* **Camelyon16**: to download the dataset, check this link :<br/>https://camelyon16.grand-challenge.org
* **MSKCC**: to download the dataset, check this link :<br/>https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52763339
* **MHIST**: to download the dataset, check this link :<br/>https://bmirds.github.io/MHIST/


## Training
The model training consists of **three** steps:
1. ***Self-supervised pretraining*** (i.e., Our earlier proposed method ***`Resolution sequence prediction (RSP)`*** and Momentum Contrast (MoCo))
2. ***Curriculum-I fine-tuning*** (`easy-to-hard`)
3. ***Curriculum-II fine-tuning*** (`hard-to-very-hard`)

### 1. Self-supervised pretraining
In this work, we build our approach on our previous work ["Self-Supervised driven Consistency Training for Annotation Efficient Histopathology Image Analysis](https://arxiv.org/abs/2102.03897). Please, refer to our previous [repository](https://github.com/srinidhiPY/SSL_CR_Histo) for pretraining details on whole-slide-images. We have included the pretrained model for Camelyon16, found in the "models" folder - cam_SSL_pretrained_model.pt.

### Fine-tuing of pretrained models on the target task using hardness-aware curriculum training
1. Download the desired pretrained model from the models folder.
2. Download the desired dataset; you can simply add any other dataset that you wish.
3. For whole-slide image (WSI) slide-level classification tasks, run the following command by the desired parameters. For example, to finetune barlowtwins on ChestX-ray14, run:
```bash
python main_classification.py --data_set ChestXray14  \
--init barlowtwins \
--proxy_dir path/to/pre-trained-model \
--data_dir path/to/dataset \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt 
```
Or, to evaluate supervised ImageNet model on ChestX-ray14, run:
```bash
python main_classification.py --data_set ChestXray14  \
--init ImageNet \
--data_dir path/to/dataset \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt 
```

