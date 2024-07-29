'''
Borrowed from PointBERT 
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from ..build import DATASETS
import torch
import logging
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@DATASETS.register_module()
class ModelNet(Dataset):
    def __init__(self,
                 data_dir, num_points, num_classes,
                 use_normals=False,
                 split='train',
                 transform=None
                 ):
        self.root = data_dir
        self.npoints = num_points
        self.use_normals = use_normals
        self.num_category = num_classes
        split = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.split = split
        f = h5py.File(os.path.join(self.root, "modelnet40normel_" + str(self.npoints) + "_" + split + ".h5"), "r")
        self.data = f['data'][:].astype(np.float32)
        self.craft = f['craft'][:].astype(np.float32)
        self.label = f['label'][:].astype(np.int64)
        logging.info('The size of %s data is %d' % (split, len(self.data)))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @property
    def num_classes(self):
        return self.num_category

    def __getitem__(self, index):
        pointcloud = np.concatenate([self.data[index], self.craft[index]], axis=-1) # [1024,6+15]
        label = self.label[index]
        if self.split == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud[:, 0:3],
                'y': label,
                'craft': pointcloud[:, 6:21]
                }
        if self.use_normals:
            data['x'] = pointcloud[:, 3:6]
        if self.transform is not None:
            data = self.transform(data)

        if self.use_normals:
            data['x'] = torch.cat((data['pos'], data['x']), dim=1)
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['x'], data['heights']), dim=1)
        if 'craft' in data.keys():
            data['x'] = torch.cat((data['x'], data['craft']), dim=1)
        return data
