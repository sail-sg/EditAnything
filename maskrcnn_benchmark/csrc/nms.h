// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}


std::pair<at::Tensor, at::Tensor> soft_nms(const at::Tensor& dets,
                                           const at::Tensor& scores,
                                           const float threshold,
                                           const float sigma) {

  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("Soft NMS Does Not have GPU support");
#endif
  }

  std::pair<at::Tensor, at::Tensor> result = soft_nms_cpu(dets, scores, threshold, sigma);

  return result;
}