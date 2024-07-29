/*
 * @Author: 20:17:19
 * @Date:   2024-01-18 20:54:24
 * @Email:  weiweijin1109@gmail.com
 */

#include <torch/extension.h>
#include "utils.hpp"

at::Tensor denew_knn_forward(at::Tensor features, at::Tensor indices, const int n);
at::Tensor denew_knn_backward(at::Tensor grad_y, at::Tensor indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &denew_knn_forward, "denew_knn forward (CUDA)");
  m.def("backward", &denew_knn_backward, "denew_knn backward (CUDA)");
}
