# -*- coding: utf-8 -*-
# @Author: Weiwei Jin
# @Date:   2024-01-18 20:17:19
# @Email:  weiweijin1109@gmail.com

import torch

import denew_knn

class Denew_knn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indices, N):
        """
        :param ctx:
        :param features: features of points, FloatTensor[B, C, M, U]
        :param indices: neighbor indices of centers, IntTensor[B, M, U], M is #centers, U is #neighbors
        :param N int
        :return:
            grouped_features: grouped features, FloatTensor[B, C, N]
        """
        features = features.contiguous()
        indices = indices.contiguous()
        ctx.save_for_backward(indices)
        return denew_knn.forward(features, indices, N)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_features = denew_knn.backward(grad_output.contiguous(), indices)
        return grad_features, None, None

denew_knn_func = Denew_knn.apply