# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import itertools

from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR, WarmupReduceLROnPlateau


def make_optimizer(cfg, model):
    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # different lr schedule
        if "language_backbone" in key:
            lr = cfg.SOLVER.LANG_LR

        if "backbone.body" in key and "language_backbone.body" not in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_BODY_LR_FACTOR

        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if 'norm' in key or 'Norm' in key:
            weight_decay *= cfg.SOLVER.WEIGHT_DECAY_NORM_FACTOR
            print("Setting weight decay of {} to {}".format(key, weight_decay))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, lr)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.MULTI_MAX_EPOCH:
        assert len(cfg.SOLVER.MULTI_MAX_EPOCH) == len(cfg.SOLVER.STEPS)
        lr_scheduler = []

        for stage_step, stage_max_epoch in zip(cfg.SOLVER.STEPS, cfg.SOLVER.MULTI_MAX_ITER):
            milestones = []
            for step in stage_step:
                milestones.append(round(step * stage_max_epoch))
            lr_scheduler.append(WarmupMultiStepLR(optimizer,
                                                  milestones,
                                                  cfg.SOLVER.GAMMA,
                                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                                  warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                                                  warmup_method=cfg.SOLVER.WARMUP_METHOD, )
                                )
        return lr_scheduler

    elif cfg.SOLVER.USE_COSINE:
        max_iters = cfg.SOLVER.MAX_ITER
        return WarmupCosineAnnealingLR(
            optimizer,
            max_iters,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            eta_min=cfg.SOLVER.MIN_LR
        )

    elif cfg.SOLVER.USE_AUTOSTEP:
        max_iters = cfg.SOLVER.MAX_ITER
        return WarmupReduceLROnPlateau(
            optimizer,
            max_iters,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            eta_min=cfg.SOLVER.MIN_LR,
            patience=cfg.SOLVER.STEP_PATIENCE,
            verbose=True
        )

    else:
        milestones = []
        for step in cfg.SOLVER.STEPS:
            if step < 1:
                milestones.append(round(step * cfg.SOLVER.MAX_ITER))
            else:
                milestones.append(step)
        return WarmupMultiStepLR(
            optimizer,
            milestones,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
