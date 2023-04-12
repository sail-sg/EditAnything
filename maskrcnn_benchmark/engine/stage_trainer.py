# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


def reduce_loss_dict(all_loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for loss_dict in all_loss_dict:
            for k in sorted(loss_dict.keys()):
                loss_names.append(k)
                all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        if world_size > 1:
            dist.reduce(all_losses, dst=0)
            if dist.get_rank() == 0:
                # only main process gets accumulated, so only divide by
                # world_size in this case
                all_losses /= world_size

        reduced_losses = {}
        for k, v in zip(loss_names, all_losses):
            if k not in reduced_losses:
                reduced_losses[k] = v / len(all_loss_dict)
            reduced_losses[k] += v / len(all_loss_dict)

    return reduced_losses


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    epoch_per_stage = arguments['epoch_per_stage']
    max_iter = sum(len(stage_loader) * epoch_per_stage[si] for si, stage_loader in enumerate(data_loader))
    max_iter += epoch_per_stage[-1] * min(len(stage_loader) for stage_loader in data_loader)
    model.train()
    start_training_time = time.time()
    end = time.time()

    for stage_i, stage_loader in enumerate(data_loader):
        for ep in range(epoch_per_stage[stage_i]):
            start_iter = arguments["iteration"]
            for iteration, (images, targets, _) in enumerate(stage_loader, start_iter):
                data_time = time.time() - end
                iteration = iteration + 1
                arguments["iteration"] = iteration

                scheduler[stage_i].step()

                all_stage_loss_dict = []
                images = images.to(device)
                targets = [target.to(device) for target in targets]
                loss_dict = model(images, targets, stage_i)
                all_stage_loss_dict.append(loss_dict)

                losses = sum(loss for loss_dict in all_stage_loss_dict for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(all_stage_loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)

                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if iteration % 20 == 0 or iteration == max_iter:
                    logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                if iteration % checkpoint_period == 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)
                if iteration == max_iter:
                    checkpointer.save("model_final", **arguments)

    for ep in range(epoch_per_stage[-1]):
        start_iter = arguments["iteration"]
        for iteration, stage_loader in enumerate(zip(*data_loader), start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            scheduler[-1].step()

            all_task_loss_dict = []
            for stage_i, (images, targets, _) in enumerate(stage_loader):
                images = images.to(device)
                targets = [target.to(device) for target in targets]
                loss_dict = model(images, targets, stage_i)
                all_task_loss_dict.append(loss_dict)

            losses = sum(loss for loss_dict in all_task_loss_dict for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(all_task_loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
