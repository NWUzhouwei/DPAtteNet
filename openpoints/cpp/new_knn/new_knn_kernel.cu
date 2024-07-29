/*
 * @Author: 20:17:19
 * @Date:   2024-01-18 20:54:24
 * @Email:  weiweijin1109@gmail.com
 */
#ifndef _NEW_KNN_KERNEL
#define _NEW_KNN_KERNEL

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "cuda_utils.cuh"
/*
  Function: new knn
  Args:
    b   : batch size
    n   : number of points in point clouds
    m   : number of set centers
    k   : number of neighbors
    centers_coords: coordinates of centers, FloatTensor[b, 3, m]
    points_coords : coordinates of points, FloatTensor[b, 3, n]
    neighbors_indices : neighbor indices in points, IntTensor[b, m, k]
    flag: flag, IntTensor[b, n]
    key: point key, IntTensor[b, n]
    dist: dist between center and other points, FloatTensor[b, m, n]
*/
__global__ void new_knn_kernel(int b, int n, int m, int k,
                                  const float * centers_coords,
                                  const float * points_coords,
                                  int * neighbors_indices,
                                  int * flag,
                                  int * key,
                                  float * dist) {
  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;
  points_coords += batch_index * n * 3;
  centers_coords += batch_index * m * 3;
  neighbors_indices += batch_index * m * k;
  flag += batch_index * n;
  key += batch_index * n;
  dist += batch_index * m * n;

  for (int i = index; i < m; i += stride) {
    float center_x = centers_coords[i];
    float center_y = centers_coords[i + m];
    float center_z = centers_coords[i + m + m];
    
    for (int j = 0; j < n; ++j) { // compute distance
      float dx = center_x - points_coords[j];
      float dy = center_y - points_coords[j + n];
      float dz = center_z - points_coords[j + n + n];
      dist[i*n + j] = dx * dx + dy * dy + dz * dz;
      key[j] = j;
    }

    //select sort (the max k)
    for (int j = 0, cnt = 0; j < n && cnt < k; ++j) {
      if (flag[key[j]] == 1) { // already be allocated
        continue;
      }

      float min_current = dist[i*n + j];
      int min_flag = 0;
      for (int u = j+1; u < n; ++u) {
        if (flag[key[u]] == 1) { // already be allocated
          continue;
        }
        if (dist[i*n + u] < min_current) {
          min_current = dist[i*n + u];
          min_flag = u;
        }
      }

      if (min_flag != 0) { // fined the minier, than exchange
        dist[i*n + min_flag] = dist[i*n + j];
        dist[i*n + j] = min_current;
        neighbors_indices[i*k + cnt] = key[min_flag];
        key[min_flag] = key[j];
        key[j] = neighbors_indices[i*k + cnt];
        flag[key[min_flag]] = 1;
        cnt++;
      } else { // current is the miniest
        neighbors_indices[i*k + cnt] = key[j];
        flag[key[j]] = 1;
        cnt++;
      }
    }
  }
}

torch::Tensor new_knn_forward(torch::Tensor centers_coords,
                           torch::Tensor points_coords){
  
  CHECK_CUDA(centers_coords);
  CHECK_CUDA(points_coords);
  CHECK_CONTIGUOUS(centers_coords);
  CHECK_CONTIGUOUS(points_coords);
  CHECK_IS_FLOAT(centers_coords);
  CHECK_IS_FLOAT(points_coords);
  
  const int b = centers_coords.size(0);
  const int m = centers_coords.size(2);
  const int n = points_coords.size(2);
  const int k = n / m;

  torch::Tensor neighbors_indices = torch::zeros(
      {b, m, k}, torch::CUDA(torch::kInt));
  torch::Tensor flag = torch::zeros(
      {b, n}, torch::CUDA(torch::kInt));
  torch::Tensor key = torch::zeros(
      {b, n}, torch::CUDA(torch::kInt));
  torch::Tensor dist = torch::zeros(
      {b, m, n}, torch::CUDA(torch::kFloat));

  new_knn_kernel<<<b, optimal_num_threads(m), 0, at::cuda::getCurrentCUDAStream()>>>(
      b, n, m, k, centers_coords.data_ptr<float>(), points_coords.data_ptr<float>(), neighbors_indices.data_ptr<int>(), 
      flag.data_ptr<int>(), key.data_ptr<int>(), dist.data_ptr<float>());

  CUDA_CHECK_ERRORS();
  return neighbors_indices;
}
#endif