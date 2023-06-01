# Things you need to know about PYSKL data format

PYSKL now provides pre-processed pickle annotations files for training and testing. The pre-processing scripts will be released in later updates. Below we demonstrate the format of the annotation files and provide the download links.

## The format of the pickle files

Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: `split` and `annotations`

1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
   1. `frame_dir` (str): The identifier of the corresponding video.
   2. `total_frames` (int): The number of frames in this video.
   3. `img_shape` (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
   4. `original_shape` (tuple[int]): Same as `img_shape`.
   5. `label` (int): The action label.
   6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
   7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

## Download the pre-processed skeletons

We provide links to the pre-processed skeleton annotations, you can directly download them and use them for training & testing.

- AffWild2 [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl


## Process AffWild2 raw skeleton files

0. Assume that you are using the current directory as the working directory, which is `$PYSKL/tools/data`
1. Follow the steps of the [face-skeleton-detection](https://github.com/shahroudy/NTURGB-D/) repositoy
2. Change the paths in the file `affwild_preproc_full.py` (details are provided within the file)
3. Run `python affwild_preproc_full.py` to generate processed skeleton annotations, it will generate `AffWild_train_full.pkl` under your current working directory.

PS: For the best pre-processing speed, change `num_process` in `affwild_preproc_full.py` to the number of cores that your CPU has.

### BibTex items for each provided dataset

```BibTex
% NTURGB+D
@inproceedings{shahroudy2016ntu,
  title={Ntu rgb+ d: A large scale dataset for 3d human activity analysis},
  author={Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1010--1019},
  year={2016}
}
% NTURGB+D 120
@article{liu2019ntu,
  title={Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding},
  author={Liu, Jun and Shahroudy, Amir and Perez, Mauricio and Wang, Gang and Duan, Ling-Yu and Kot, Alex C},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={10},
  pages={2684--2701},
  year={2019},
  publisher={IEEE}
}
% Kinetics-400
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
% GYM
@inproceedings{shao2020finegym,
  title={Finegym: A hierarchical video dataset for fine-grained action understanding},
  author={Shao, Dian and Zhao, Yue and Dai, Bo and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2616--2625},
  year={2020}
}
% UCF101
@article{soomro2012ucf101,
  title={UCF101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}
% HMDB51
@inproceedings{kuehne2011hmdb,
  title={HMDB: a large video database for human motion recognition},
  author={Kuehne, Hildegard and Jhuang, Hueihan and Garrote, Est{\'\i}baliz and Poggio, Tomaso and Serre, Thomas},
  booktitle={2011 International conference on computer vision},
  pages={2556--2563},
  year={2011},
  organization={IEEE}
}
% Diving48
@inproceedings{li2018resound,
  title={Resound: Towards action recognition without representation bias},
  author={Li, Yingwei and Li, Yi and Vasconcelos, Nuno},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={513--528},
  year={2018}
}
```
