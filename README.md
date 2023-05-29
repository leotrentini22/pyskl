# PYSKL

PYSKL is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. Various algorithms will be supported for skeleton-based action recognition. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

This repo is an adaption of [PoseConv3D](https://arxiv.org/abs/2104.13586) on the face action units recognition task

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Data Preparation

We provide HRNet 2D skeletons for every dataset we supported and Kinect 3D skeletons for the NTURGB+D and NTURGB+D 120 dataset. To obtain the human skeleton annotations, you can:

1. Use our pre-processed skeleton annotations: we directly provide the processed skeleton data for all datasets as pickle files (which can be directly used for training and testing), check [Data Doc](/tools/data/README.md) for the download links and descriptions of the annotation format.
2. For NTURGB+D 3D skeletons, you can download the official annotations from https://github.com/shahroudy/NTURGB-D, and use our [provided script](/tools/data/ntu_preproc.py) to generate the processed pickle files. The generated files are the same with the provided `ntu60_3danno.pkl` and `ntu120_3danno.pkl`. For detailed instructions, follow the [Data Doc](/tools/data/README.md).
3. We also provide scripts to extract 2D HRNet skeletons from RGB videos, you can follow the [diving48_example](/examples/extract_diving48_skeleton/diving48_example.ipynb) to extract 2D skeletons from an arbitrary RGB video dataset.

You can use [vis_skeleton](/demo/vis_skeleton.ipynb) to visualize the provided skeleton data.

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
For specific examples, please go to the README for each specific algorithm we supported.

