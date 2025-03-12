import os
import glob
import h5py
import json
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@DATASETS.register_module()
class ShapeNetPartNormal(Dataset):
    classes = ['airplane', 'bag', 'cap', 'car', 'chair',
               'earphone', 'guitar', 'knife', 'lamp', 'laptop',
               'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]


    cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
                   'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
                   'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
                   'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}
    cls2parts = []
    cls2partembed = torch.zeros(16, 50)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat

    def __init__(self,
                 data_root='data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 num_points=2048,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=16,
                 presample=False,
                 sampler='fps', 
                 transform=None,
                 multihead=False,
                 **kwargs
                 ):
        self.npoints = num_points
        self.root = data_root
        self.transform = transform
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.use_normal = use_normal
        self.presample = presample
        self.sampler = sampler 
        self.split = split
        self.multihead=multihead
        self.part_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if (
                    (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(self.cat[item] + '\\' + token + '.txt')

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        if transform is None:
            self.eye = np.eye(shape_classes)
        else:
            self.eye = torch.eye(shape_classes)

        # in the testing, using the uniform sampled 2048 points as input
        # presample
        filename = os.path.join(data_root, 'processed',
                                f'{split}_{num_points}_fps.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data, self.cls = [], []
            npoints = []
            for cat, filepath in tqdm(self.datapath, desc=f'Sample ShapeNetPart {split} split'):
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int64)
                data_path = os.path.join(self.root, filepath)
                craft_path = os.path.join(self.root, 'craft', filepath)
                craft = np.loadtxt(craft_path).astype(np.float32)
                data = np.loadtxt(data_path).astype(np.float32)
                data = np.concatenate([data, craft], axis=-1)
                npoints.append(len(data))
                data = torch.from_numpy(data).to(
                    torch.float32).cuda().unsqueeze(0)
                data = fps(data, num_points).cpu().numpy()[0]
                self.data.append(data)
                self.cls.append(cls)
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(os.path.join(data_root, 'processed'), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump((self.data, self.cls), f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data, self.cls = pickle.load(f)
                print(f"{filename} load successfully")

    def __getitem__(self, index):
        if not self.presample:
            fn = self.datapath[index][1]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data_path = os.path.join(self.root, fn)
            craft_path = os.path.join(self.root, 'craft', fn)
            data = np.loadtxt(data_path).astype(np.float32)
            craft = np.loadtxt(craft_path).astype(np.float32) # [n,15]
            point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int64)
        else:
            data, cls = self.data[index], self.cls[index]
            point_set, seg = data[:, :6], data[:, 6].astype(np.int64)
            craft = data[:, 7:22]

        if 'train' in self.split:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice]
            craft = craft[choice]
            seg = seg[choice]
        else:
            point_set = point_set[:self.npoints]
            seg = seg[:self.npoints]
        if self.multihead:
            seg=seg-self.part_start[cls[0]]

        data = {'pos': point_set[:, 0:3],
                'x': point_set[:, 3:6],
                'craft': craft,
                'cls': cls,
                'y': seg}

        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
        """
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    train = ShapeNetPartNormal(num_points=2048, split='trainval')
    test = ShapeNetPartNormal(num_points=2048, split='test')
    for dict in train:
        for i in dict:
            print(i, dict[i].shape)