

# Unpaired-Cross-modality-Image-Synthesis

This is our PyTorch implementation for Unpaired whole-body MR to CT Image Synthesis. It is still under active development.

The code was written by [Yunhao Ge](https://github.com/gyhandy) based on the structure of Cycle-GAN ([Jun-Yan Zhu](https://github.com/junyanz)).

**Unpaired Whole-body MR to CT Synthesis with Correlation Coefficient Constrained Adversarial Learning: [PDF](https://github.com/gyhandy/publication/raw/master/Unpaired%20whole-body%20MR%20to%20CT%20synthesis%20with%20correlation%20coefficient%20constrained%20adversarial%20learning-SPIE.pdf) 

## Abstract

MR to CT image synthesis plays an important role in medical image analysis, and its applications included, but not limited to PET-MR attenuation correction and MR only radiation therapy planning.Recently, deep learning-based image synthesis techniques have achieved much success. However, most of the current methods require large scales of paired data from two different modalities, which greatly limits their usage as in some situation paired data is infeasible to obtain. Some efforts have been proposed to relax this constraint such as cycle-consistent adversarial networks (Cycle-GAN). However, the cycle consistency loss is an indirect structural similarity constraint of input and synthesized images, and it sometimes lead to inferior synthesized results.  
Contribution  
1 Proposed an explicit structural constrained adversarial learning method to improve both the realistic and precise of the synthesized images which were unique to cross-modality medical image mapping  
2 Designed a novel correlation coefficient loss, which directly constrained the structural similarity between the input Magnetic Resonance (MR) and synthesized Computed Tomography (CT) image, to solve the mismatch of anatomical structures in synthesized CT images  
3 Developed a shape discriminator to incorporate the shape consistency information by extracting shape masks from two modality images
to improve the synthesis quality. Gained substantial quality improvement especially in the surface shape and bone in whole body image
mapping  

#### original_data
<img src='imgs/show1.png' width="800px">

#### pipeline

<img src='imgs/show5.png' width="800px">
<img src='imgs/Show21.png' width="800px">

#### performance

<img src='imgs/show3.png' width="800px">
<img src='imgs/show4.png' width="800px">

## Code description

### data

Medical raw data preprocess and dataset class

- Turn the 3D slice Medical image to 2D slice and satisfy the data structure of algorithm，`dataset_pre_new.py`
- Dataset containing (MR, CT, MR_mask) when training，`unaligned_dataset.py`
- Sigle dataset containing MR only when testing/mapping，`single_dataset.py`
- Basic dataset class of Cycle-GAN，`base_dataset.py`

### models

Models define and structures

- Basic model structure，`base_model.py`
- Explicit constraint adversarial learning model based on Cycle-GAN，`cycle_gan_model.py`
- Mapping model when testing，`test_model.py`
- Basic class and networks，`networks.py`

### options

Parameter settings when training and testing

### datasets

- test original data from one patient with 4 modality MR images and unpaired CT images，`ZS18158187`
- training dataset with trainA(MR), trainB(CT), maskA(MR_mask), testA(MR) and testB（CT)`MR2CT`

### CT_segmentation

Training a shape extractor for our explicit constraint adversarial learning

### Unet

Shape extractor defination of structure of Unet

### checkpoints

Our trained model with unpaired whole-body MR and CT images

- our best performance model with our method，`Self+Lcc_finetune`
- trained model with only correlation coefficient loss，`cycle_Lcc`

### output

output of test-adapt or test_pipeline


### fill_hole.py

Prepare training data for MR and CT mask 

### train.py

Train a new model with our explicit constraint adversarial learning

### test-adapt.py

Make a image synthesis with our trained model on one patient
(We have already made the data preprocess, and you can find the synthesised data in output)

### test-pipeline.py

Make image synthesis with our trained model on multiple data



## Getting Started
### Installation
- Install PyTorch 0.4, torchvision, and other dependencies from http://pytorch.org
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom dominate
```
- Alternatively, all dependencies can be installed by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```
- For Conda users, we include a script `./scripts/conda_deps.sh` to install PyTorch and other libraries.

###  train a explicit constraint model on MR to CT mapping

For the dataset in training and testing, please create a new document 'datasets' and download the datasets in it.(link：https://pan.baidu.com/s/17Q8XERh3Qo5aPXGgpGhZfA 
code：57rm ; If the expiry date is exceeded, please email gyhandy@sjtu.edu.cn )

```bash
python train.py
```
The trained model will be saved to : `./checkpoints/{model_name}`.

###  test our trained model

```bash
python test-adapt.py
```
The test results will be saved to : `./output/{model_name}`.


## cite

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)
In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)


