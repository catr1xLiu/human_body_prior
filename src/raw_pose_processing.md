## Extract Poses from Amass Dataset


```python
%load_ext autoreload
%autoreload 2
%matplotlib notebook
%matplotlib inline

import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm



from human_body_prior.tools.omni_tools import copy2cpu as c2c

os.environ['PYOPENGL_PLATFORM'] = 'egl'
```

    Failed to reload module 'packaging._structures' from file '/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/packaging/_structures.py'
    Traceback (most recent call last):
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 325, in check
        superreload(m, reload, self.old_objects)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 625, in superreload
        update_generic(old_obj, new_obj)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 451, in update_generic
        update(a, b)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 419, in update_class
        update_instances(old, new)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 377, in update_instances
        object.__setattr__(ref, "__class__", new)
    TypeError: __class__ assignment: 'InfinityType' object layout differs from 'InfinityType'


    [autoreload of packaging._structures failed: Traceback (most recent call last):
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 325, in check
        superreload(m, reload, self.old_objects)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 625, in superreload
        update_generic(old_obj, new_obj)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 451, in update_generic
        update(a, b)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 419, in update_class
        update_instances(old, new)
      File "/home/catr1x/Coding/Human Motion Model/Code/LoRA-MDM-Age-Dataset/human_body_prior/.venv/lib64/python3.12/site-packages/IPython/extensions/autoreload.py", line 377, in update_instances
        object.__setattr__(ref, "__class__", new)
    TypeError: __class__ assignment: 'InfinityType' object layout differs from 'InfinityType'
    ]


    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


### Please remember to download the following subdataset from AMASS website: https://amass.is.tue.mpg.de/download.php. Note only download the <u>SMPL+H G</u> data.
* ACCD (ACCD)
* HDM05 (MPI_HDM05)
* TCDHands (TCD_handMocap)
* SFU (SFU)
* BMLmovi (BMLmovi)
* CMU (CMU)
* Mosh (MPI_mosh)
* EKUT (EKUT)
* KIT  (KIT)
* Eyes_Janpan_Dataset (Eyes_Janpan_Dataset)
* BMLhandball (BMLhandball)
* Transitions (Transitions_mocap)
* PosePrior (MPI_Limits)
* HumanEva (HumanEva)
* SSM (SSM_synced)
* DFaust (DFaust_67)
* TotalCapture (TotalCapture)
* BMLrub (BioMotionLab_NTroje)

### Unzip all datasets. In the bracket we give the name of the unzipped file folder. Please correct yours to the given names if they are not the same.

### Place all files under the directory **./amass_data/**. The directory structure shoud look like the following:  
./amass_data/  
./amass_data/ACCAD/  
./amass_data/BioMotionLab_NTroje/  
./amass_data/BMLhandball/  
./amass_data/BMLmovi/   
./amass_data/CMU/  
./amass_data/DFaust_67/  
./amass_data/EKUT/  
./amass_data/Eyes_Japan_Dataset/  
./amass_data/HumanEva/  
./amass_data/KIT/  
./amass_data/MPI_HDM05/  
./amass_data/MPI_Limits/  
./amass_data/MPI_mosh/  
./amass_data/SFU/  
./amass_data/SSM_synced/  
./amass_data/TCD_handMocap/  
./amass_data/TotalCapture/  
./amass_data/Transitions_mocap/  

**Please make sure the file path are correct, otherwise it can not succeed.**


```python
# Choose the device to run the body model on.
comp_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
```


```python
from human_body_prior.body_model.body_model import BodyModel

male_bm_path = './body_models/smplh/male/model.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

female_bm_path = './body_models/smplh/female/model.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
faces = c2c(male_bm.f)

female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)
```


```python
paths = []
folders = []
dataset_names = []
for root, dirs, files in os.walk('./amass_data'):
#     print(root, dirs, files)
#     for folder in dirs:
#         folders.append(os.path.join(root, folder))
    folders.append(root)
    for name in files:
        dataset_name = root.split('/')[2]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, name))
```


```python
save_root = './pose_data'
save_folders = [folder.replace('./amass_data', './pose_data') for folder in folders]
for folder in save_folders:
    os.makedirs(folder, exist_ok=True)
group_path = [[path for path in paths if name in path] for name in dataset_names]
```


```python
trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
ex_fps = 20
def amass_to_pose(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True)
    fps = 0
    try:
        fps = bdata['mocap_framerate']
        frame_number = bdata['trans'].shape[0]
    except:
#         print(list(bdata.keys()))
        return fps
    
    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    if bdata['gender'] == 'male':
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / ex_fps)
#     print(frame_number)
#     print(fps)

    bdata_poses = bdata['poses'][::down_sample,...]
    bdata_trans = bdata['trans'][::down_sample,...]
    body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(comp_device),
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(comp_device),
            'trans': torch.Tensor(bdata_trans).to(comp_device),
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)).to(comp_device),
        }
    
    with torch.no_grad():
        body = bm(**body_parms)
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    
    
    np.save(save_path, pose_seq_np_n)
    return fps
```


```python
group_path = group_path
all_count = sum([len(paths) for paths in group_path])
cur_count = 0
```

This will take a few hours for all datasets, here we take one dataset as an example

To accelerate the process, you could run multiple scripts like this at one time.


```python
import time
for paths in group_path:
    dataset_name = paths[0].split('/')[2]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s'%dataset_name)
    fps = 0
    for path in pbar:
        save_path = path.replace('./amass_data', './pose_data')
        save_path = save_path[:-3] + 'npy'
        fps = amass_to_pose(path, save_path)
        
    cur_count += len(paths)
    print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
    time.sleep(0.5)
```

    Processing: SFU: 100%|██████████| 44/44 [11:42<00:00, 15.97s/it]


    Processed / All (fps 120): 44/44


The above code will extract poses from **AMASS** dataset, and put them under directory **"./pose_data"**

The source data from **HumanAct12** is already included in **"./pose_data"** in this repository. You need to **unzip** it right in this folder.

## Segment, Mirror and Relocate Motions


```python
import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
```


```python
def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data
```


```python
index_path = './index.csv'
save_dir = './joints'
index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]
fps = 20
```


```python
for i in tqdm(range(total_amount)):
    source_path = index_file.loc[i]['source_path']
    new_name = index_file.loc[i]['new_name']
    data = np.load(source_path)
    start_frame = index_file.loc[i]['start_frame']
    end_frame = index_file.loc[i]['end_frame']
    if 'humanact12' not in source_path:
        if 'Eyes_Japan_Dataset' in source_path:
            data = data[3*fps:]
        if 'MPI_HDM05' in source_path:
            data = data[3*fps:]
        if 'TotalCapture' in source_path:
            data = data[1*fps:]
        if 'MPI_Limits' in source_path:
            data = data[1*fps:]
        if 'Transitions_mocap' in source_path:
            data = data[int(0.5*fps):]
        data = data[start_frame:end_frame]
        data[..., 0] *= -1
    
    data_m = swap_left_right(data)
#     save_path = pjoin(save_dir, )
    np.save(pjoin(save_dir, new_name), data)
    np.save(pjoin(save_dir, 'M'+new_name), data_m)
```

    100%|██████████| 14616/14616 [00:30<00:00, 486.52it/s]



```python

```
