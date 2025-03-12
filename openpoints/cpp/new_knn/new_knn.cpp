/*
 * @Author: 20:17:19
 * @Date:   2024-01-18 20:54:24
 * @Email:  weiweijin1109@gmail.com
 */
#ifndef _NEW_KNN
#define _NEW_KNN
#include <torch/extension.h>

/**
    neighbors_indices : neighbor indices in points, IntTensor[b, m, k]
    flag: flag, IntTensor[b, n]
    key: point key, IntTensor[b, n]
    dist: dist between center and other points, FloatTensor[b, m, n]
*/
torch::Tensor new_knn_forward(torch::Tensor centers_coords,
                              torch::Tensor points_coords);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &new_knn_forward, "New KNN (CUDA)");
}
#endif