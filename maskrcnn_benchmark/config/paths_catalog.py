# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


def try_to_find(file, return_dir=False, search_path=['./DATASET', './OUTPUT', './data', './MODEL']):
    if not file:
        return file

    if file.startswith('catalog://'):
        return file

    DATASET_PATH = ['./']
    if 'DATASET' in os.environ:
        DATASET_PATH.append(os.environ['DATASET'])
    DATASET_PATH += search_path

    for path in DATASET_PATH:
        if os.path.exists(os.path.join(path, file)):
            if return_dir:
                return path
            else:
                return os.path.join(path, file)

    print('Cannot find {} in {}'.format(file, DATASET_PATH))
    exit(1)


class DatasetCatalog(object):
    DATASETS = {
        # pretrained grounding dataset
        # mixed vg and coco
        "mixed_train": {
            "coco_img_dir": "coco/train2014",
            "vg_img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/final_mixed_train.json",
        },
        "mixed_train_no_coco": {
            "coco_img_dir": "coco/train2014",
            "vg_img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/final_mixed_train_no_coco.json",
        },

        # flickr30k
        "flickr30k_train": {
            "img_folder": "flickr30k/flickr30k_images/train",
            "ann_file": "mdetr_annotations/final_flickr_separateGT_train.json",
            "is_train": True
        },
        "flickr30k_val": {
            "img_folder": "flickr30k/flickr30k_images/val",
            "ann_file": "mdetr_annotations/final_flickr_separateGT_val.json",
            "is_train": False
        },
        "flickr30k_test": {
            "img_folder": "flickr30k/flickr30k_images/test",
            "ann_file": "mdetr_annotations/final_flickr_separateGT_test.json",
            "is_train": False
        },

        # refcoco
        "refexp_all_val": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/final_refexp_val.json",
            "is_train": False
        },

        # gqa
        "gqa_val": {
            "img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/final_gqa_val.json",
            "is_train": False
        },

        # phrasecut
        "phrasecut_train": {
            "img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/finetune_phrasecut_train.json",
            "is_train": True
        },


        # od to grounding
        # coco tsv
        "coco_dt_train": {
            "dataset_file": "coco_dt",
            "yaml_path": "coco_tsv/coco_obj.yaml",
            "is_train": True,
        },
        "COCO_odinw_train_8copy_dt_train": {
            "dataset_file": "coco_odinw_dt",
            "yaml_path": "coco_tsv/COCO_odinw_train_8copy.yaml",
            "is_train": True,
        },
        "COCO_odinw_val_dt_train": {
            "dataset_file": "coco_odinw_dt",
            "yaml_path": "coco_tsv/COCO_odinw_val.yaml",
            "is_train": False,
        },
        # lvis tsv
        "lvisv1_dt_train": {
            "dataset_file": "lvisv1_dt",
            "yaml_path": "coco_tsv/LVIS_v1_train.yaml",
            "is_train": True,
        },
        "LVIS_odinw_train_8copy_dt_train": {
            "dataset_file": "coco_odinw_dt",
            "yaml_path": "coco_tsv/LVIS_odinw_train_8copy.yaml",
            "is_train": True,
        },
        # object365 tsv
        "object365_dt_train": {
            "dataset_file": "object365_dt",
            "yaml_path": "Objects365/objects365_train_vgoiv6.cas2000.yaml",
            "is_train": True,
        },
        "object365_odinw_2copy_dt_train": {
            "dataset_file": "object365_odinw_dt",
            "yaml_path": "Objects365/objects365_train_odinw.cas2000_2copy.yaml",
            "is_train": True,
        },
        "objects365_odtsv_train": {
            "dataset_file": "objects365_odtsv",
            "yaml_path": "Objects365/train.cas2000.yaml",
            "is_train": True,
        },
        "objects365_odtsv_val": {
            "dataset_file": "objects365_odtsv",
            "yaml_path": "Objects365/val.yaml",
            "is_train": False,
        },

        # ImagetNet OD
        "imagenetod_train_odinw_2copy_dt": {
            "dataset_file": "imagenetod_odinw_dt",
            "yaml_path": "imagenet_od/imagenetod_train_odinw_2copy.yaml",
            "is_train": True,
        },

        # OpenImage OD
        "oi_train_odinw_dt": {
            "dataset_file": "oi_odinw_dt",
            "yaml_path": "openimages_v5c/oi_train_odinw.cas.2000.yaml",
            "is_train": True,
        },

        # vg tsv
        "vg_dt_train": {
            "dataset_file": "vg_dt",
            "yaml_path": "visualgenome/train_vgoi6_clipped.yaml",
            "is_train": True,
        },

        "vg_odinw_clipped_8copy_dt_train": {
            "dataset_file": "vg_odinw_clipped_8copy_dt",
            "yaml_path": "visualgenome/train_odinw_clipped_8copy.yaml",
            "is_train": True,
        },
        "vg_vgoi6_clipped_8copy_dt_train": {
            "dataset_file": "vg_vgoi6_clipped_8copy_dt",
            "yaml_path": "visualgenome/train_vgoi6_clipped_8copy.yaml",
            "is_train": True,
        },

        # coco json
        "coco_grounding_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json",
            "is_train": True,
        },

        "lvis_grounding_train": {
            "img_dir": "coco",
            "ann_file": "coco/annotations/lvis_od_train.json"
        },


        "lvis_val": {
            "img_dir": "coco",
            "ann_file": "coco/annotations/lvis_od_val.json"
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2017_test": {
            "img_dir": "coco/test2017",
            "ann_file": "coco/annotations/image_info_test-dev2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
    }

    @staticmethod
    def set(name, info):
        DatasetCatalog.DATASETS.update({name: info})

    @staticmethod
    def get(name):

        if name.endswith('_bg'):
            attrs = DatasetCatalog.DATASETS[name]
            data_dir = try_to_find(attrs["ann_file"], return_dir=True)
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="Background",
                args=args,
            )
        else:
            if "bing" in name.split("_"):
                attrs = DatasetCatalog.DATASETS["bing_caption_train"]
            else:
                attrs = DatasetCatalog.DATASETS[name]

            if "voc" in name and 'split' in attrs:
                data_dir = try_to_find(attrs["data_dir"], return_dir=True)
                args = dict(
                    data_dir=os.path.join(data_dir, attrs["data_dir"]),
                    split=attrs["split"],
                )
                return dict(
                    factory="PascalVOCDataset",
                    args=args,
                )
            elif "mixed" in name:
                vg_img_dir = try_to_find(attrs["vg_img_dir"], return_dir=True)
                coco_img_dir = try_to_find(attrs["coco_img_dir"], return_dir=True)
                ann_file = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder_coco=os.path.join(coco_img_dir, attrs["coco_img_dir"]),
                    img_folder_vg=os.path.join(vg_img_dir, attrs["vg_img_dir"]),
                    ann_file=os.path.join(ann_file, attrs["ann_file"])
                )
                return dict(
                    factory="MixedDataset",
                    args=args,
                )
            elif "flickr" in name:
                img_dir = try_to_find(attrs["img_folder"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_folder"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                    is_train=attrs["is_train"]
                )
                return dict(
                    factory="FlickrDataset",
                    args=args,
                )
            elif "refexp" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="RefExpDataset",
                    args=args,
                )
            elif "gqa" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="GQADataset",
                    args=args,
                )
            elif "phrasecut" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="PhrasecutDetection",
                    args=args,
                )
            elif "_caption" in name:
                yaml_path = try_to_find(attrs["yaml_path"], return_dir=True)
                if "no_coco" in name:
                    yaml_name = attrs["yaml_name_no_coco"]
                else:
                    yaml_name = attrs["yaml_name"]
                yaml_file_name = "{}.{}.yaml".format(yaml_name, name.split("_")[2])
                args = dict(
                    yaml_file=os.path.join(yaml_path, attrs["yaml_path"], yaml_file_name)
                )
                return dict(
                    factory="CaptionTSV",
                    args=args,
                )
            elif "inferencecap" in name:
                yaml_file_name = try_to_find(attrs["yaml_path"])
                args = dict(
                    yaml_file=yaml_file_name)
                return dict(
                    factory="CaptionTSV",
                    args=args,
                )
            elif "pseudo_data" in name:
                args = dict(
                    yaml_file=try_to_find(attrs["yaml_path"])
                )
                return dict(
                    factory="PseudoData",
                    args=args,
                )
            elif "_dt" in name:
                dataset_file = attrs["dataset_file"]
                yaml_path = try_to_find(attrs["yaml_path"], return_dir=True)
                args = dict(
                    name=dataset_file,
                    yaml_file=os.path.join(yaml_path, attrs["yaml_path"]),
                )
                return dict(
                    factory="CocoDetectionTSV",
                    args=args,
                )
            elif "_odtsv" in name:
                dataset_file = attrs["dataset_file"]
                yaml_path = try_to_find(attrs["yaml_path"], return_dir=True)
                args = dict(
                    name=dataset_file,
                    yaml_file=os.path.join(yaml_path, attrs["yaml_path"]),
                )
                return dict(
                    factory="ODTSVDataset",
                    args=args,
                )
            elif "_grounding" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="CocoGrounding",
                    args=args,
                )
            elif "lvis_evaluation" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="LvisDetection",
                    args=args,
                )
            else:
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                args = dict(
                    root=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                for k, v in attrs.items():
                    args.update({k: os.path.join(ann_dir, v)})
                return dict(
                    factory="COCODataset",
                    args=args,
                )

        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/20171220/X-101-64x4d": "ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
