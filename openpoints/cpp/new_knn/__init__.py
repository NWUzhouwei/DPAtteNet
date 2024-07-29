# -*- coding: utf-8 -*-
# @Author: Weiwei Jin
# @Date:   2024-01-18 20:17:19
# @Email:  weiweijin1109@gmail.com

import torch
import new_knn

def new_knn_func(centers_coords, points_coords):
        """
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :return:
            neighbor_indices: indices of neighbors, IntTensor[B, M, K] K = N / M
        """
        centers_coords = centers_coords.contiguous()
        points_coords = points_coords.contiguous()
        return new_knn.forward(centers_coords, points_coords)
