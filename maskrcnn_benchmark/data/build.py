# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import os

import torch.utils.data
import torch.distributed as dist
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms

from transformers import AutoTokenizer
from .datasets.duplicate_dataset import create_duplicate_dataset

def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True, class_concat=False, extra_args={}):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    num_category = 1
    for dataset_id, dataset_name in enumerate(dataset_list, 1):
        if is_train:
            dataset_name = dataset_name + cfg.DATASETS.TRAIN_DATASETNAME_SUFFIX
        else:
            dataset_name = dataset_name + cfg.DATASETS.TEST_DATASETNAME_SUFFIX
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train

        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        if data["factory"] in ["VGTSVDataset", "CocoDetectionTSV", "ODTSVDataset"]:
            args["extra_fields"] = ["class"]
            if cfg.MODEL.MASK_ON:
                args["extra_fields"].append("mask")

        if data["factory"] in ["CocoGrounding", "CocoDetectionTSV", "CaptionTSV", "MixedDataset", "FlickrDataset", "RefExpDataset", "GQADataset", "PseudoData", "PhrasecutDetection"]:
            # args["return_masks"] = False
            args["return_masks"] = cfg.MODEL.MASK_ON
            args["return_tokens"] = True
            args["max_num_labels"] = cfg.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM
            args["max_query_len"] = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN

        args["transforms"] = transforms
        args.update(extra_args)

        if dataset_name == "flickr30k_train":
            copy = cfg.DATASETS.FLICKR_COPY
        elif dataset_name in ["mixed_train", "mixed_train_no_coco"]:
            copy = cfg.DATASETS.MIXED_COPY
        elif dataset_name == "COCO_odinw_train_8copy_dt_train":
            copy = cfg.DATASETS.COCO_COPY
        elif dataset_name == "LVIS_odinw_train_8copy_dt_train":
            copy = cfg.DATASETS.LVIS_COPY
        elif dataset_name == "object365_odinw_2copy_dt_train":
            copy = cfg.DATASETS.OBJECT365_COPY
        elif dataset_name == "vg_odinw_clipped_8copy_dt_train":
            copy = cfg.DATASETS.VG_COPY
        elif dataset_name == "vg_vgoi6_clipped_8copy_dt_train":
            copy = cfg.DATASETS.VG_COPY
        elif dataset_name == "imagenetod_train_odinw_2copy_dt":
            copy = cfg.DATASETS.IN_COPY
        elif dataset_name == "oi_train_odinw_dt":
            copy = cfg.DATASETS.OI_COPY
        elif is_train:
            copy = cfg.DATASETS.GENERAL_COPY
        elif not is_train:
            copy = cfg.DATASETS.GENERAL_COPY_TEST
        else:
            copy = -1 # do not ever copy test
        
        if copy != -1:
            new_factory = create_duplicate_dataset(factory)
            dataset = new_factory(copy=copy, **args)
        else:
            # make dataset from factory
            dataset = factory(**args)

        print(dataset_name, 'has the {} data points'.format(len(dataset)), data["factory"])

        if class_concat:
            category = list(dataset.contiguous_category_id_to_json_id.values())
            dataset.contiguous_category_id_to_json_id = {}
            dataset.json_category_id_to_contiguous_id = {}
            for id, cat in enumerate(category, start=num_category):
                dataset.json_category_id_to_contiguous_id[cat] = id
                dataset.contiguous_category_id_to_json_id[id] = cat
            num_category += len(category)
            print("Found {} #category after group {}, concating ...".format(num_category, dataset_id))
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def build_dataset_by_group(dataset_list, transforms, dataset_catalog, is_train=True, class_by_group=True,
                           class_concat=False, extra_args={}):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )

    num_category = 1
    grouped_datasets = []
    for group_id, group in enumerate(dataset_list, 1):
        datasets = []
        for dataset_name in group:
            data = dataset_catalog.get(dataset_name)
            factory = getattr(D, data["factory"])
            args = data["args"]
            # for COCODataset, we want to remove images without annotations
            # during training
            if data["factory"] == "COCODataset":
                args["remove_images_without_annotations"] = is_train
            if data["factory"] == "PascalVOCDataset":
                args["use_difficult"] = not is_train
            args["transforms"] = transforms
            args.update(extra_args)
            # make dataset from factory
            dataset = factory(**args)

            # check if dataset is grouped by task, assume one class per task
            if class_by_group and data["factory"] != "Background":
                category = dataset.contiguous_category_id_to_json_id[1]
                del dataset.contiguous_category_id_to_json_id[1]
                dataset.json_category_id_to_contiguous_id[category] = group_id
                dataset.contiguous_category_id_to_json_id[group_id] = category

            datasets.append(dataset)

        if class_concat:
            for dataset in datasets:
                category = list(dataset.contiguous_category_id_to_json_id.values())
                dataset.contiguous_category_id_to_json_id = {}
                dataset.json_category_id_to_contiguous_id = {}
                for id, cat in enumerate(category, start=num_category):
                    dataset.json_category_id_to_contiguous_id[cat] = id
                    dataset.contiguous_category_id_to_json_id[id] = cat
            num_category += len(category)
            print("Found {} #category after group {}, concating ...".format(num_category, group_id))

        if is_train:
            datasets = D.ConcatDataset(datasets)

        grouped_datasets.append(datasets)

    # for testing, return a list of datasets
    if not is_train:
        datasets = [dataset for group in grouped_datasets for dataset in group]
        return datasets
    if class_concat:
        grouped_datasets = D.ConcatDataset(grouped_datasets)
        return [grouped_datasets]

    # for training, concatenate all datasets into a single one
    return grouped_datasets


def make_data_sampler(dataset, shuffle, distributed, num_replicas=None, rank=None, use_random_seed=True):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank,
                                           use_random=use_random_seed)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0, drop_last=False
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=drop_last
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=drop_last
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_data_loader(cfg, is_train=True, is_distributed=False, num_replicas=None, rank=None, start_iter=0):
    num_gpus = num_replicas or get_world_size()

    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )

    DatasetCatalog = paths_catalog.DatasetCatalog
    if len(cfg.DATASETS.REGISTER) > 0:
        for new_dataset in cfg.DATASETS.REGISTER:
            # img_dir = cfg.DATASETS.REGISTER[new_dataset]["img_dir"]
            # if "ann_file" in cfg.DATASETS.REGISTER[new_dataset]:
            #     ann_file = cfg.DATASETS.REGISTER[new_dataset]["ann_file"]
            # else:
            #     ann_file = None
            attrs = dict(cfg.DATASETS.REGISTER[new_dataset])
            if is_train:
                new_dataset = new_dataset + cfg.DATASETS.TRAIN_DATASETNAME_SUFFIX
            else:
                new_dataset = new_dataset + cfg.DATASETS.TEST_DATASETNAME_SUFFIX
            DatasetCatalog.set(new_dataset, attrs)


    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    # Haotian: expand bing dataset
    if "bing_caption_train" in dataset_list and len(cfg.DATASETS.BING_INDEX_LIST) > 0:
        dataset_list = list(dataset_list)
        dataset_list.remove("bing_caption_train")
        for bing_index in cfg.DATASETS.BING_INDEX_LIST:
            dataset_list.insert(len(dataset_list), "bing_caption_{}_train".format(bing_index))
        dataset_list = tuple(dataset_list)
    
    if "bing_caption_train_no_coco" in dataset_list and len(cfg.DATASETS.BING_INDEX_LIST) > 0:
        dataset_list = list(dataset_list)
        dataset_list.remove("bing_caption_train_no_coco")
        for bing_index in cfg.DATASETS.BING_INDEX_LIST:
            dataset_list.insert(len(dataset_list), "bing_caption_{}_train_no_coco".format(bing_index))
        dataset_list = tuple(dataset_list)

    print("The combined datasets are: {}.".format(dataset_list))

    transforms = None if not is_train and cfg.TEST.USE_MULTISCALE else build_transforms(cfg, is_train)

    extra_args = {}
    if is_train and cfg.DATASETS.USE_CROWD:
        extra_args['ignore_crowd'] = False
    if is_train and cfg.DATASETS.MAX_BOX > 0:
        extra_args['max_box'] = cfg.DATASETS.MAX_BOX
    if is_train and cfg.DATASETS.FEW_SHOT>0:
        extra_args['few_shot'] = cfg.DATASETS.FEW_SHOT
    if is_train and cfg.DATASETS.SHUFFLE_SEED != 0:
        extra_args['shuffle_seed'] = cfg.DATASETS.SHUFFLE_SEED

    # od to grounding
    if is_train and cfg.DATASETS.RANDOM_SAMPLE_NEG > 0:
        extra_args['random_sample_negative'] = cfg.DATASETS.RANDOM_SAMPLE_NEG
    if is_train and cfg.DATASETS.ADD_DET_PROMPT:
        extra_args["add_detection_prompt"] = True
    if is_train and cfg.DATASETS.USE_OD_AUG:
        extra_args["use_od_data_aug"] = True
    if is_train and cfg.DATASETS.DISABLE_SHUFFLE:
        extra_args["disable_shuffle"] = True
    if cfg.DATASETS.ONE_HOT:
        extra_args["one_hot"] = True
    if is_train and len(cfg.DATASETS.PROMPT_VERSION) > 0:
        extra_args["prompt_engineer_version"] = cfg.DATASETS.PROMPT_VERSION
    if is_train and len(cfg.DATASETS.CONTROL_PROB) == 4:
        extra_args["control_probabilities"] = cfg.DATASETS.CONTROL_PROB
    if is_train and cfg.DATASETS.DISABLE_CLIP_TO_IMAGE:
        extra_args["disable_clip_to_image"] =  cfg.DATASETS.DISABLE_CLIP_TO_IMAGE
    if is_train and cfg.DATASETS.NO_MINUS_ONE_FOR_ONE_HOT:
        extra_args["no_minus_one_for_one_hot"] = cfg.DATASETS.NO_MINUS_ONE_FOR_ONE_HOT
    if is_train:
        extra_args["separation_tokens"] = cfg.DATASETS.SEPARATION_TOKENS
    # caption
    if is_train and cfg.DATASETS.CAPTION_MIN_BOX > 0:
        extra_args["caption_min_box"] = cfg.DATASETS.CAPTION_MIN_BOX
    if is_train and cfg.DATASETS.REPLACE_CLEAN_LABEL:
        extra_args["replace_clean_label"] = True
    if is_train and cfg.DATASETS.FURTHER_SCREEN:
        extra_args["further_screen"] = True
    if is_train and cfg.DATASETS.CAPTION_CONF > 0.0:
        extra_args["caption_conf"] = cfg.DATASETS.CAPTION_CONF
    if is_train:
        extra_args["caption_nms"] = cfg.DATASETS.CAPTION_NMS
    if is_train and cfg.DATASETS.PACK_RANDOM_CAPTION_NUMBER > 0:
        extra_args["pack_random_caption_number"] = cfg.DATASETS.PACK_RANDOM_CAPTION_NUMBER
    if is_train and cfg.DATASETS.INFERENCE_CAPTION:
        extra_args["inference_caption"] = True
    if is_train and cfg.DATASETS.SAMPLE_NEGATIVE_FOR_GROUNDING_DATA > 0:
        extra_args["sample_negative_for_grounding_data"] = cfg.DATASETS.SAMPLE_NEGATIVE_FOR_GROUNDING_DATA
    if is_train and cfg.DATASETS.RANDOM_PACK_PROB > 0:
        extra_args["random_pack_prob"] = cfg.DATASETS.RANDOM_PACK_PROB
    if is_train and cfg.DATASETS.NO_RANDOM_PACK_PROBABILITY > 0:
        extra_args["no_random_pack_probability"] = cfg.DATASETS.NO_RANDOM_PACK_PROBABILITY
    if is_train:
        extra_args["safeguard_positive_caption"] = cfg.DATASETS.SAFEGUARD_POSITIVE_CAPTION
    if is_train:
        extra_args["local_debug"] = cfg.DATASETS.LOCAL_DEBUG
    if is_train:
        extra_args["no_mask_for_od"] = cfg.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_OD
    if is_train:
        extra_args["no_mask_for_gold"] = cfg.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_GOLD
    if is_train:
        extra_args["mlm_obj_for_only_positive"] = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_OBJ_FOR_ONLY_POSITIVE
    if cfg.DATASETS.OVERRIDE_CATEGORY and cfg.DATASETS.USE_OVERRIDE_CATEGORY:
        extra_args["override_category"] = cfg.DATASETS.OVERRIDE_CATEGORY
    if is_train:
        extra_args["caption_format_version"] = cfg.DATASETS.CAPTION_FORMAT_VERSION
    if is_train:
        extra_args["special_safeguard_for_coco_grounding"] = cfg.DATASETS.SPECIAL_SAFEGUARD_FOR_COCO_GROUNDING
    if is_train:
        extra_args["diver_box_for_vqa"] = cfg.DATASETS.DIVER_BOX_FOR_VQA
    extra_args["caption_prompt"] = cfg.DATASETS.CAPTION_PROMPT
    extra_args["use_caption_prompt"] = cfg.DATASETS.USE_CAPTION_PROMPT

    # extra_args['tokenizer'] = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
        # extra_args['tokenizer'] = build_tokenizer("clip")
        from transformers import CLIPTokenizerFast
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            extra_args["tokenizer"] = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", from_slow=True, mask_token='ðŁĴĳ</w>')
        else:
            extra_args["tokenizer"] = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", from_slow=True)
    else:
        extra_args['tokenizer'] = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)

    if isinstance(dataset_list[0], (tuple, list)):
        datasets = build_dataset_by_group(dataset_list, transforms, DatasetCatalog, is_train,
                                          class_by_group=cfg.DATASETS.ALTERNATIVE_TRAINING,
                                          class_concat=cfg.DATASETS.CLASS_CONCAT,
                                          extra_args=extra_args)
    else:
        datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train,
                                 class_concat=cfg.DATASETS.CLASS_CONCAT,
                                 extra_args=extra_args)

    data_loaders = []
    for di, dataset in enumerate(datasets):
        if is_train and cfg.SOLVER.MAX_EPOCH > 0:
            num_iters = cfg.SOLVER.MAX_EPOCH * len(dataset) // cfg.SOLVER.IMS_PER_BATCH
            print("Number of iterations are {}".format(num_iters))
            cfg.defrost()
            cfg.SOLVER.MAX_ITER = num_iters
            cfg.SOLVER.DATASET_LENGTH = len(dataset)
            cfg.freeze()
        if is_train and cfg.SOLVER.MULTI_MAX_EPOCH:
            num_iters = None
            cfg.defrost()
            cfg.SOLVER.MULTI_MAX_ITER += (cfg.SOLVER.MULTI_MAX_EPOCH[di] * len(dataset) // cfg.SOLVER.IMS_PER_BATCH,)
            cfg.freeze()

        if is_train and cfg.DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE:
            from .datasets.custom_distributed_sampler import DistributedSamplerChunkByNode
            chunk_or_not = []
            for i in dataset_list:
                if "bing_caption" in i:
                    chunk_or_not.append(True)
                else:
                    chunk_or_not.append(False)
            assert(len(chunk_or_not) == len(dataset.datasets))
            '''
            If we are training on 4 nodes, each with 8 GPUs
            '''
            num_nodes = int(os.getenv('NODE_COUNT', os.getenv('OMPI_COMM_WORLD_SIZE', 1)))
            local_size = cfg.num_gpus//num_nodes
            node_rank = int(os.getenv('NODE_RANK', os.getenv('OMPI_COMM_WORLD_RANK', 0)))
            local_rank = cfg.local_rank
            sampler = DistributedSamplerChunkByNode(
                dataset = dataset,
                all_datasets = dataset.datasets, # Assumming dataset is a ConcateDataset instance,
                chunk_or_not = chunk_or_not,
                num_replicas = cfg.num_gpus, # total GPU number, e.g., 32
                rank = dist.get_rank(), # Global Rank, e.g., 0~31
                node_rank = node_rank, # Node Rank, e.g., 0~3
                node_number = num_nodes, # how many node e.g., 4
                process_num_per_node = local_size, # e.g., 8
                rank_within_local_node = local_rank, # e.g., 0~7
            )
        else:
            sampler = make_data_sampler(dataset, shuffle, is_distributed, num_replicas=num_replicas, rank=rank,
                                        use_random_seed=cfg.DATALOADER.USE_RANDOM_SEED)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter, drop_last=is_train
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.USE_MULTISCALE else BatchCollator(
            cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train and cfg.SOLVER.MULTI_MAX_EPOCH:
        cfg.defrost()
        cfg.SOLVER.MULTI_MAX_ITER += (
            cfg.SOLVER.MULTI_MAX_EPOCH[-1] * min([len(dataset) // cfg.SOLVER.IMS_PER_BATCH for dataset in datasets]),)
        cfg.freeze()

    if is_train and not cfg.DATASETS.ALTERNATIVE_TRAINING and not cfg.DATASETS.MULTISTAGE_TRAINING:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]

    return data_loaders
