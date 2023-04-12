// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor ml_nms(const at::Tensor& dets,
                  const at::Tensor& scores,
                  const at::Tensor& labels,
                  const float threshold) {

  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1), labels.unsqueeze(1)}, 1);
    return ml_nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implemented");
}
