# Adaptive Teacher Weighting for Multi-Teacher Distillation with Soft Actor-Critic and Hierarchical Design

This project provides the source code of Multi-Teacher Knowledge Distillation with with Soft Actor-Critic and Hierarchical Design (MKD-RLs):

## Installation

### Requirements

Ubuntu 20.04 LTS

Python 3.9

CUDA 11.8

you can install python packages by: 

```
pip install -r requirements.txt
```

## Perform experiments on CIFAR-100 dataset

### Dataset

CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to your dataset folder

### Training teacher networks

```
python train_baseline.py --model [model_name] \
    --data-folder [your dataset path] \
    --checkpoint-dir [your checkpoint saved path]
```
* `--model`: specify the teacher model
* `--data-folder`: specify the dataset folder
* `--checkpoint-dir`: specify checkpoints folder

configure the `setting.py` with `teacher_name:teacher_path`, then you can use these pretained models in subsequent experiments.


### Training the student network with Multi-teacher KD

#### MTKD-RL

```
python train_student_rl.py \
    --data [your dataset path] \
    --arch [your student name] \
    --dynamic \
    --checkpoint-dir [your checkpoint saved path] \
    --teacher-name-list [your teacher names, separated by spaces] \
    --rank 0 
```

* `--data`: specify the dataset folder
* `--arch`: specify the student architecture, such as `MobileNetV2`
* `--dynamic`: specify whether using dynamic weight aggregation strategy. 
* `--checkpoint-dir`: specify checkpoints folder
* `--teacher-name-list`: specify the teacher names to construct the teacher pool, e.g. `RegNetY_400MF`, `RegNetX_400MF`, `resnet32x4`





#### MTKD-SAC and MTKD-HP (Ours)

```
python train_student_sac.py \
    --data [your dataset path] \
    --arch [your student name] \
    --checkpoint-dir [your checkpoint saved path] \
    --teacher-name-list [your teacher names, separated by spaces] \
    --rank 0 
```
`train_student_sac.py` can be replaced by other python files: 
* `train_student_hp.py` adds high-level output





