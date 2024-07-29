/*
 * @Author: 20:17:19
 * @Date:   2024-01-18 20:54:24
 * @Email:  weiweijin1109@gmail.com
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.hpp"
#include "cuda_utils.cuh"

/*
  Function: denew_knn features of neighbors (backward)
  Args:
    b   : batch size
    c   : #channles of features
    n   : number of points in point clouds
    m   : number of query centers
    u   : maximum number of neighbors
    grad_y: points' features, FloatTensor[b, c, n]
    indices : neighbor indices in points, IntTensor[b, m, u]
    grad_x     : gathered features, FloatTensor[b, c, m, u]
*/
__global__ void denew_knn_grad_kernel(int b, int c, int n, int m, int u,
                                const float *__restrict__ grad_y,
                                const int *__restrict__ indices,
                                float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  grad_y += batch_index * n * c;
  indices += batch_index * m * u;
  grad_x += batch_index * m * u * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * m; i += stride) {
    const int l = i / m;
    const int j = i % m;
    for (int k = 0; k < u; ++k) {
      grad_x[(l * m + j) * u + k] = grad_y[l * n + indices[j * u + k]];
    }
  }
}

at::Tensor denew_knn_backward(at::Tensor grad_y, at::Tensor indices) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int n = grad_y.size(2);
  int m = indices.size(1);
  int u = indices.size(2);
  at::Tensor grad_x = torch::zeros(
      {b, c, m, u}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  
  denew_knn_grad_kernel<<<b, optimal_block_config(m, c), 0, at::cuda::getCurrentCUDAStream()>>>
      (b, c, n, m, u, grad_y.data_ptr<float>(), indices.data_ptr<int>(), grad_x.data_ptr<float>());
  
  CUDA_CHECK_ERRORS();
  return grad_x;
}

/*
  Function: denew_knn features of neighbors (forward)
  Args:
    b   : batch size
    c   : #channles of features
    n   : number of points in point clouds
    m   : number of query centers
    u   : maximum number of neighbors
    features : grad of gathered features, FloatTensor[b, c, m, u]
    indices : neighbor indices in points, IntTensor[b, m, u]
    output: grad of points' features, FloatTensor[b, c, n]
*/
__global__ void denew_knn_kernel(int b, int c, int n, int m, int u,
                                     const float *__restrict__ features,
                                     const int *__restrict__ indices,
                                     float *__restrict__ output) {
  int batch_index = blockIdx.x;
  features += batch_index * m * u * c;
  indices += batch_index * m * u;
  output += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * m; i += stride) {
    const int l = i / m;
    const int j = i % m;
    for (int k = 0; k < u; ++k) {
      atomicAdd(output + l * n + indices[j * u + k],
                features[(l * m + j) * u + k]);
    }
  }
}

at::Tensor denew_knn_forward(at::Tensor features, at::Tensor indices, const int n) {
  CHECK_CUDA(features);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(indices);

  int b = features.size(0);
  int c = features.size(1);
  int m = indices.size(1);
  int u = indices.size(2);
  at::Tensor output = torch::zeros(
      {b, c, n}, at::device(features.device()).dtype(at::ScalarType::Float));
  
  denew_knn_kernel<<<b, optimal_block_config(m, c), 0, at::cuda::getCurrentCUDAStream()>>>
    (b, c, n, m, u, features.data_ptr<float>(), indices.data_ptr<int>(), output.data_ptr<float>());
  
  CUDA_CHECK_ERRORS();
  return output;
}
