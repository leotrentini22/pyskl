<div id="top"></div>

<br />
<div align="center">
<h1 align="center">Learning Facial Action Unit Recognition through a general action recognition algorithm</h1>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#General-Information">General Information</a></li>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Data-and-preprocessing">Data and preprocessing</a></li>
    <li><a href="#Configuration">Configuration</a></li>
    <li><a href="#Training-and-Testing">Training and Testing</a></li>
    <li><a href="#Main-Results">Main Results</a></li>
  </ol>
</details>

## General Information

The repository contains the code for the second part of the Facial Action Unit Recognition project. In particular, this repository is an adaption of [PoseConv3D](https://arxiv.org/abs/2104.13586), a general action recognition algorithm that we get from PYSKL, on the face action units recognition task.

PYSKL is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. It supports various algorithms for skeleton-based action recognition. PYSKL is based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Data and preprocessing
The Dataset we used:
  * [AffWild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)

To obtain the facial skeleton annotations, you can:

1. Use our pre-processed skeleton annotations: we directly provide the processed skeleton data as pickle files (which can be directly used for training and testing), check [Data Doc](/tools/data/README.md) for the download links and descriptions of the annotation format.
2. As an alternative,  use our [provided script](/tools/data/affwild_preproc_full.py) to generate the processed pickle files. The generated file is the same with the provided `AffWild_train_full.pkl`. For detailed instructions, follow the [Data Doc](/tools/data/README.md).

You can modify [vis_skeleton](/demo/vis_skeleton.ipynb) to visualize the skeleton data.

## Configuration
Before running, please modify the configuration file [configuration file](/configs/posec3d/slowonly_r50_affwild_xsub/joint.py) with your own personal paths (and preferences)


## Training and Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh configs/posec3d/slowonly_r50_affwild_xsub/joint.py ${NUM_GPUS} --validate --test-last --test-best
# Testing
bash tools/dist_test.sh configs/posec3d/slowonly_r50_affwild_xsub/joint.py ${CHECKPOINT_FILE} ${NUM_GPUS} --eval top_k_accuracy mean_class_accuracy --out result.pkl
```

We provide a release of our trained model. If you want to use it, please download it and run the commands above by setting the correct path of the model after `--resume`

### Our trained models

AffWild2
|arch_type|GoogleDrive link| Average F1-score|
| :--- | :---: |  :---: |
|`Ours (ResNet-50)`| [link](https://drive.google.com/file/d/1gYVHRjIj6ounxtTdXMje4WNHFMfCTVF9/view?usp=sharing) | 48.28 |


## Main Results

As a final result, we obtained an average f1-score of **48.23** on the test set

**AffWild2**

|   Method  | AU1 | AU2 | AU4 | AU6 | AU7 | AU10 | AU12 | AU15 | AU23 | AU24 | AU25 | AU26 | Avg. |
| :-------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   Ours (ResNet-50) | 56.35 | 35.54 | 49.45 | 58.97 | 73.61 | 74.03 | 69.59 | 32.47 | 14.76 | 8.77 | 84.09 | 23.97 | 48.47 |

<p align="right">(<a href="#top">Back to top</a>)</p>




