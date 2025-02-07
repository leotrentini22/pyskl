import copy as cp
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
from mmcv import dump
from tqdm import tqdm
import json

# ON FULL DATASET

#from pyskl.smp import mrlines

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

eps = 1e-3

def parse_keypoints(json_path, joints, score):
    try:        
        # Parse the JSON string
        with open(json_path) as f:
            data = json.load(f)
        keypoints = data[0]['keypoints']

        # Iterate over the elements in the keypoints array
        for i in range(0, len(keypoints), 3):
            # Split the string into a list of three strings
            doublet = keypoints[i:i+2]
            v = keypoints[i+2:i+3]
            # Remove commas from each string
            #triplet = [val.strip(',') for val in triplet]
            # Convert each string to a float
            x, y = doublet #[float(val) for val in triplet]
            # Assign the resulting values to the joints array
            joints[i//3] = [x, y]
            score[i//3] = v

        return joints, score
    except (ValueError, IndexError, KeyError) as e:
        # If an exception is raised, return the original joints array
        print('error in parse keypoints for the following file', flush=True)
        print(ske_file, flush=True)
        return joints, score


def parse_skeleton_file(ske_name, root): #='/home/trentini/face-skeleton-detection/data/AffWild2/skeletons/Train_Set'):
    body_data = dict()
    ske_file = osp.join(root, ske_name) # + '.predictions.json')
    try:    
        with open(ske_file) as f:
            data = json.load(f)
        
        keypoints = data[0]['keypoints']

        lines = mrlines(ske_file)
        idx = 0
        num_frames = 1 #int(lines[0])
        num_joints = 133
        #idx += 1

        fidx = 0

        for f in range(num_frames):
            num_bodies = 1 #int(lines[idx])
            #idx += 1
            if num_bodies == 0:
                continue
            for b in range(num_bodies):
                bodyID = 0 #int(lines[idx].split()[0])
                if bodyID not in body_data:
                    kpt = []
                    kpt_score = []
                    body_data[bodyID] = dict(kpt=kpt, start=fidx, kpt_score=kpt_score)
                
                #idx += 1
                #assert int(lines[idx]) == 90
                #idx += 1
                joints = np.zeros((num_joints, 2), dtype=np.float32)
                score = np.zeros((num_joints, 1), dtype=np.float32)
                joints, score = parse_keypoints(ske_file, joints, score)
                if joints is None:
                    return None

                # for j in range(num_joints):
                #     line = lines[idx].split()
                #     print(line, flush=True)
                #     joints[j, :3] = np.array(line[:3], dtype=np.float32)
                #     print(joints[j, :3], flush=True)
                #     idx += 1
                body_data[bodyID]['kpt'].append(joints)
                body_data[bodyID]['kpt_score'].append(score)
            fidx += 1

        for k in body_data:
            body_data[k]['motion'] = np.sum(np.var(np.vstack(body_data[k]['kpt']), axis=0))
            body_data[k]['kpt'] = np.stack(body_data[k]['kpt'])
            body_data[k]['kpt_score'] = np.stack(body_data[k]['kpt_score'])

        #assert idx == len(lines)
        return body_data
    except (ValueError, IndexError, KeyError) as e:
        # If an exception is raised, return the original joints array
        print('error in parse_skeleton_files for the following file', flush=True)
        print(ske_file, flush=True)
        return body_data


def spread_denoising(body_data_list):
    wh_ratio = 0.8
    spnoise_ratio = 0.69754

    def get_valid_frames(kpt):
        valid_frames = []
        for i in range(kpt.shape[0]):
            x, y = kpt[i, :, 0], kpt[i, :, 1]
            if (x.max() - x.min()) <= wh_ratio * (y.max() - y.min()):
                valid_frames.append(i)
        return valid_frames

    for item in body_data_list:
        valid_frames = get_valid_frames(item['kpt'])
        if len(valid_frames) == item['kpt'].shape[0]:
            item['flag'] = True
            continue
        ratio = len(valid_frames) / item['kpt'].shape[0]
        if 1 - ratio >= spnoise_ratio:
            item['flag'] = False
        else:
            item['flag'] = True
            item['motion'] = min(item['motion'],
                                 np.sum(np.var(item['kpt'][valid_frames].reshape(-1, 3), axis=0)))
    body_data_list = [item for item in body_data_list if item['flag']]
    assert len(body_data_list) >= 1
    _ = [item.pop('flag') for item in body_data_list]
    body_data_list.sort(key=lambda x: -x['motion'])
    return body_data_list


def non_zero(kpt):
    s = 0
    e = kpt.shape[1]
    while np.sum(np.abs(kpt[:, s])) < eps:
        s += 1
    while np.sum(np.abs(kpt[:, e - 1])) < eps:
        e -= 1
    return kpt[:, s: e]


def gen_keypoint_array(body_data):
    length_threshold = 11

    body_data = cp.deepcopy(list(body_data.values()))
    body_data.sort(key=lambda x: -x['motion'])
    if len(body_data) == 1:
        return body_data[0]['kpt'][None]
    else:
        body_data = [item for item in body_data if item['kpt'].shape[0] > length_threshold]
        if len(body_data) == 1:
            return body_data[0]['kpt'][None]
        body_data = spread_denoising(body_data)
        if len(body_data) == 1:
            return body_data[0]['kpt'][None]
        max_fidx = 0

        for item in body_data:
            max_fidx = max(max_fidx, item['start'] + item['kpt'].shape[0])
        keypoint = np.zeros((2, max_fidx, 25, 3), np.float32)

        s1, e1, s2, e2 = body_data[0]['start'], body_data[0]['start'] + body_data[0]['kpt'].shape[0], 0, 0
        keypoint[0, s1: e1] = body_data[0]['kpt']
        for item in body_data[1:]:
            s, e = item['start'], item['start'] + item['kpt'].shape[0]
            if max(s1, s) >= min(e1, e):
                keypoint[0, s: e] = item['kpt']
                s1, e1 = min(s, s1), max(e, e1)
            elif max(s2, s) >= min(e2, e):
                keypoint[1, s: e] = item['kpt']
                s2, e2 = min(s, s2), max(e, e2)

        keypoint = non_zero(keypoint)
        if np.sum(np.abs(keypoint[0, 0, 1])) < eps and np.sum(np.abs(keypoint[1, 0, 1])) > eps:
            keypoint = keypoint[::-1]
        return keypoint
    
def gen_keypoint_score_array(body_data):
    length_threshold = 11

    body_data = cp.deepcopy(list(body_data.values()))
    body_data.sort(key=lambda x: -x['motion'])
    if len(body_data) == 1:
        return body_data[0]['kpt_score'][None]
    else:
        body_data = [item for item in body_data if item['kpt_score'].shape[0] > length_threshold]
        if len(body_data) == 1:
            return body_data[0]['kpt_score'][None]
        body_data = spread_denoising(body_data)
        if len(body_data) == 1:
            return body_data[0]['kpt_score'][None]
        max_fidx = 0

        for item in body_data:
            max_fidx = max(max_fidx, item['start'] + item['kpt'].shape[0])
        keypoint = np.zeros((2, max_fidx, 25, 3), np.float32)

        s1, e1, s2, e2 = body_data[0]['start'], body_data[0]['start'] + body_data[0]['kpt_score'].shape[0], 0, 0
        keypoint[0, s1: e1] = body_data[0]['kpt_score']
        for item in body_data[1:]:
            s, e = item['start'], item['start'] + item['kpt_score'].shape[0]
            if max(s1, s) >= min(e1, e):
                keypoint[0, s: e] = item['kpt_score']
                s1, e1 = min(s, s1), max(e, e1)
            elif max(s2, s) >= min(e2, e):
                keypoint[1, s: e] = item['kpt_score']
                s2, e2 = min(s, s2), max(e, e2)

        keypoint = non_zero(keypoint)
        if np.sum(np.abs(keypoint[0, 0, 1])) < eps and np.sum(np.abs(keypoint[1, 0, 1])) > eps:
            keypoint = keypoint[::-1]
        return keypoint

def gen_anno(name, labels):
    body_data = parse_skeleton_file(name, root)
    if len(body_data) == 0:
        return None
    keypoint = gen_keypoint_array(body_data).astype(np.float16)
    label = labels 
    total_frames = 1 #keypoint.shape[1]
    img_shape = (256,256)
    original_shape = img_shape
    keypoint_score = gen_keypoint_score_array(body_data).astype(np.float16)

    return dict(frame_dir=name, label=label, keypoint=keypoint, total_frames=total_frames, img_shape=img_shape, original_shape=original_shape, keypoint_score=keypoint_score)

def get_labels(it, root):
    #returns labels (i.e. AUs) of the image
    index = root.index("skeletons")
    path = root[0:index]
    set = os.path.basename(root)
    if (set == 'Train_Set'):
        with open(os.path.join(path,'list/AffWild2_train_label.txt'), 'r') as f:
            lines = f.readlines()
    elif (set == 'Validation_Set'):
        with open(os.path.join(path,'list/AffWild2_val_label.txt'), 'r') as f:
            lines = f.readlines()
    else:
        with open(os.path.join(path,'list/AffWild2_test_label.txt'), 'r') as f:
            lines = f.readlines()
    # it+1 because it has to skip the first line, line 0 doesn't count
    return np.array(lines[it+1].split('\n')[0].split()).astype(float)

def get_json(it, root):
    #returns the json path
    index = root.index("skeletons")
    path = root[0:index]
    set = os.path.basename(root)
    if (set == 'Train_Set'):
        with open(os.path.join(path,'list/AffWild2_train_img_path.txt'), 'r') as f:
            lines = f.readlines()
    elif (set == 'Validation_Set'):
        with open(os.path.join(path,'list/AffWild2_val_img_path.txt'), 'r') as f:
            lines = f.readlines()
    else:
        with open(os.path.join(path,'list/AffWild2_test_img_path.txt'), 'r') as f:
            lines = f.readlines()
    # it+1 because it has to skip the first line, line 0 doesn't count
    image_path = lines[it+1]
    directory_name = os.path.basename(os.path.dirname(image_path))
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    json_file_name = f"{directory_name}.{file_name}.predictions.json"
    return json_file_name


### Here starts the script

### HERE CHANGE THE PATH WHERE YOU HAVE ALL THE SKELETONS
path = '/home/trentini/face-skeleton-detection/data/AffWild2/skeletons'

### THIS IS BECAUSE I HAVE LISTED 41 AUS, BUT AFFWILD2 HAS ONLY 12 AUS, I KEEP ONLY THE ONES OF INTEREST
positions_to_keep = [0, 1, 2, 4, 5, 7, 9, 12, 19, 20, 21, 22]


### Train

set = 'Train_Set'
root = os.path.join(path, set)
skeleton_files = os.listdir(root)
names_train = skeleton_files 

anno_dict = {}
num_process = 1
file_count=[]

if num_process == 1:
    # Each annotations has 7 keys: frame_dir, label, keypoint, total_frames, img_shape, original_shape, keypoint_score
    it = 0
    for name in tqdm(range(100000)): # MORE OR LESS 400k json in train dataset (seems not complete dataset)
        labels = get_labels(it, root)
        labels = labels[positions_to_keep]
        file_json = get_json(it, root)
        diction = gen_anno(file_json, labels)
        anno_dict[file_json] = diction
        file_count.append(file_json)
        it += 1
else:
    names_train = skeleton_files 
    pool = mp.Pool(num_process)
    annotations = pool.map(gen_anno, names_train)
    pool.close()
    for anno in annotations:
        anno_dict[anno['frame_dir']] = anno

names_train = [x for x in names_train if anno_dict is not None]
#xsub_train = [skeleton_files[it] for it in range(200) if file_count[it] is not None]


### Val

set = 'Validation_Set'
root = os.path.join(path, set)
skeleton_files = os.listdir(root)
names_val = skeleton_files 


if num_process == 1:
    # Each annotations has 7 keys: frame_dir, label, keypoint, total_frames, img_shape, original_shape, keypoint_score
    it = 0
    for name in tqdm(range(30000)): # MORE OR LESS 250k json in validation dataset (seems not complete dataset)
        labels = get_labels(it, root)
        labels = labels[positions_to_keep]
        file_json = get_json(it, root)
        anno_dict[file_json] = gen_anno(file_json, labels)
        file_count.append(file_json)
        it += 1
else:
    names_val = skeleton_files 
    pool = mp.Pool(num_process)
    annotations = pool.map(gen_anno, names_val)
    pool.close()
    for anno in annotations:
        anno_dict[anno['frame_dir']] = anno

names_val = [x for x in names_val if anno_dict is not None]
# xsub_val = [skeleton_files[it] for it in range(200) if file_count[it] is not None]


xsub_train = [name for name in names_train]
xsub_val = [name for name in names_val]
# xview_train = [name for name in names if 'C001' not in name]
# xview_val = [name for name in names if 'C001' in name]
split = dict(xsub_train=xsub_train, xsub_val=xsub_val) #xview_train=xview_train, xview_val=xview_val)
annotations = [anno_dict[name] for name in file_count] #names]
dump(dict(split=split, annotations=annotations), 'AffWild_train_full.pkl')
