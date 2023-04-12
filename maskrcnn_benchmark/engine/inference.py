# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import re

import torch
from tqdm import tqdm
from collections import defaultdict

from maskrcnn_benchmark.data.datasets.evaluation import evaluate, im_detect_bbox_aug
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize
import pdb
from maskrcnn_benchmark.data.datasets.evaluation.flickr.flickr_eval import FlickrEvaluator
from maskrcnn_benchmark.structures.bounding_box import BoxList
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
def imshow(img, file_name = "tmp.jpg"):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    #plt.figtext(0.5, 0.09, "test", wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(file_name)
def load(url_or_file_name):
    try:
        response = requests.get(url_or_file_name)
    except:
        response = None
    if response is None:
        pil_image = Image.open(url_or_file_name).convert("RGB")
    else:
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image
def inference_default(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        cfg=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, *_ = batch
        with torch.no_grad():
            if cfg.TEST.USE_MULTISCALE:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    predictions = results_dict
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return None

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args)


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def create_one_hot_dict(labels, no_minus_one_for_one_hot = False):
    positive_map_token_to_label = defaultdict(int)
    positive_map_label_to_token = defaultdict(int)

    for i in range(len(labels)):
        positive_map_token_to_label[i] = labels[i]
        positive_map_label_to_token[labels[i]] = i

    if no_minus_one_for_one_hot:
        positive_map_token_to_label = defaultdict(int)
        positive_map_label_to_token = defaultdict(int)

        for i in range(len(labels)):
            positive_map_token_to_label[i+1] = labels[i]
            positive_map_label_to_token[labels[i]] = i + 1

    return positive_map_token_to_label, positive_map_label_to_token


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
    positive_map = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = defaultdict(list)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]  # because the labels starts from 1
                positive_map_label_to_token[labels[j]].append(i)
            # positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map, positive_map_label_to_token  # / (positive_map.sum(-1)[:, None] + 1e-6)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert(counter == len(lst))

    return all_

def create_queries_and_maps_from_dataset(dataset, cfg):
    categories = dataset.categories()
    #one_hot = dataset.one_hot

    labels = []
    label_list = []
    keys = list(categories.keys())
    keys.sort()
    for i in keys:
        labels.append(i)
        label_list.append(categories[i])

    if cfg.TEST.CHUNKED_EVALUATION != -1:
        labels = chunks(labels, cfg.TEST.CHUNKED_EVALUATION)
        label_list = chunks(label_list, cfg.TEST.CHUNKED_EVALUATION)
    else:
        labels = [labels]
        label_list = [label_list]

    all_queries = []
    all_positive_map_label_to_token = []

    for i in range(len(labels)):
        labels_i = labels[i]
        label_list_i = label_list[i]
        query_i, positive_map_label_to_token_i = create_queries_and_maps(
            labels_i, label_list_i, additional_labels = cfg.DATASETS.SUPRESS_QUERY if cfg.DATASETS.USE_SUPRESS_QUERY else None, cfg = cfg)
        
        all_queries.append(query_i)
        all_positive_map_label_to_token.append(positive_map_label_to_token_i)
    print("All queries", all_queries)
    return all_queries, all_positive_map_label_to_token

def create_queries_and_maps(labels, label_list, additional_labels = None, cfg = None):

    # Clean label list
    original_label_list = label_list.copy()
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""

    # sep between tokens, follow training
    separation_tokens = cfg.DATASETS.SEPARATION_TOKENS
    
    caption_prompt = cfg.DATASETS.CAPTION_PROMPT
    if caption_prompt is not None and isinstance(caption_prompt, str):
        caption_prompt = load_from_yaml_file(caption_prompt)
    use_caption_prompt = cfg.DATASETS.USE_CAPTION_PROMPT and caption_prompt is not None
    for _index, label in enumerate(label_list):
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["prefix"]
        
        start_i = len(objects_query)

        if use_caption_prompt:
            objects_query += caption_prompt[_index]["name"]
        else:
            objects_query += label
        
        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]
        
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["suffix"]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens
    
    if additional_labels is not None:
        objects_query += separation_tokens
        for _index, label in enumerate(additional_labels):
            objects_query += label
            if _index != len(additional_labels) - 1:
                objects_query += separation_tokens

    print(objects_query)

    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenized = tokenizer(objects_query, return_tensors="pt")
    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
        from transformers import CLIPTokenizerFast
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True, mask_token='ðŁĴĳ</w>')
        else:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True)
        tokenized = tokenizer(objects_query,
                              max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                              truncation=True,
                              return_tensors="pt")
    else:
        tokenizer = None
        raise NotImplementedError

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(tokenized, tokens_positive,
                                                                                        labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token

def create_positive_map_label_to_token_from_positive_map(positive_map, plus = 0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token



def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def resize_box(output, targets):
    if isinstance(targets[0], dict):
        orig_target_sizes = targets[0]["orig_size"].unsqueeze(0)
    else:
        orig_target_sizes = torch.stack([targets[0].extra_fields["orig_size"] for _ in range(1)], dim=0)
    img_h, img_w = orig_target_sizes.unbind(1)
    return output.resize((img_w, img_h))

def flickr_post_process(output, targets, positive_map_label_to_token, plus):
    output = resize_box(output, targets)
    scores, indices = torch.topk(output.extra_fields["scores"], k = len(output.extra_fields["scores"]), sorted=True)
    boxes = output.bbox.tolist()
    boxes = [boxes[i] for i in indices]
    labels = [output.extra_fields["labels"][i] for i in indices]
    output_boxes = [[] for i in range(len(positive_map_label_to_token))]
    output_scores = [[] for i in range(len(positive_map_label_to_token))]
    for i in range(len(boxes)):
        output_boxes[labels[i] - plus].append(boxes[i])
        output_scores[labels[i] - plus].append(scores[i])
    for i in output_boxes:
        i.append([0.0, 0.0, 0.0, 0.0])
    image_ids = [t.extra_fields["original_img_id"] for t in targets]
    sentence_ids = [t.extra_fields["sentence_id"] for t in targets]

    return {"image_id": image_ids[0], "sentence_id": sentence_ids[0], "boxes": output_boxes, "scores": output_scores}

def build_flickr_evaluator(cfg):
    evaluator = FlickrEvaluator(
        "DATASET/flickr30k/flickr30k/", # Hard written!!
        subset="test" if "test" in cfg.DATASETS.TEST[0]  else "val",
        merge_boxes=cfg.DATASETS.FLICKR_GT_TYPE == "merged")
    return evaluator

def build_lvis_evaluator(ann_file, fixed_ap=True):
    from maskrcnn_benchmark.data.datasets.evaluation.lvis.lvis import LVIS
    from maskrcnn_benchmark.data.datasets.evaluation.lvis.lvis_eval import LvisEvaluatorFixedAP, LvisEvaluator
    evaluator = LvisEvaluatorFixedAP(LVIS(ann_file), fixed_ap=fixed_ap)
    #evaluator = LvisEvaluator(LVIS(ann_file), iou_types=['segm', 'bbox'])
    return evaluator

def write_lvis_results(results, output_file_name):
    lines = []
    lines.append("metric, avg ")
    for each_result in results:
        metric_string = " ".join(each_result.split(" ")[:-2])
        number = each_result.split(" ")[-1]
        each_result = metric_string + ", " + number + " "
        lines.append(each_result)

    string_to_write = "\n".join(lines) + "\n"
    with open(output_file_name, "w") as f:
        f.write(string_to_write)
    return

def write_flickr_results(results, output_file_name):
    '''
    {'Recall@1_all': 0.8394651146677753, 'Recall@1_animals': 0.9177820267686424, 'Recall@1_bodyparts': 0.7097966728280961, ...}
    '''
    lines = []
    lines.append("metric, avg ")
    for each_metric, number in results.items():
        each_result = each_metric + ", " + str(number) + " "
        lines.append(each_result)

    string_to_write = "\n".join(lines) + "\n"
    with open(output_file_name, "w") as f:
        f.write(string_to_write)
    return

def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        cfg=None,
        verbose=True,
        visualizer = None
):
    # convert to a torch.device for efficiency
    try:
        device = torch.device(device)
    except:
        device = device
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    if verbose:
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    task = cfg.TEST.EVAL_TASK

    if not task:
        return inference_default(model, data_loader, dataset_name, iou_types, box_only, device, expected_results, expected_results_sigma_tol, output_folder, cfg)
        
    if cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
        assert task == 'detection'
        categories = dataset.categories()

        keys = list(categories.keys())
        keys.sort()
        all_queries = [[categories[k] for k in keys]]
        all_positive_map_label_to_token = [{k: [i] for i, k in enumerate(keys)}]
    elif task == "detection":
        all_queries, all_positive_map_label_to_token = create_queries_and_maps_from_dataset(dataset, cfg)
    elif task == "grounding":
        all_queries = [None]
        all_positive_map_label_to_token = [None]
    else:
        assert(0)

    '''
    Build Dataset Sepecific Evaluator
    '''
    if "flickr" in cfg.DATASETS.TEST[0]:
        evaluator = build_flickr_evaluator(cfg)
    elif "lvis" in cfg.DATASETS.TEST[0]:
        evaluator = build_lvis_evaluator(dataset.ann_file, fixed_ap=not cfg.DATASETS.LVIS_USE_NORMAL_AP)
    else:
        evaluator = None

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    if verbose:
        _iterator = tqdm(data_loader)
    else:
        _iterator = data_loader
    for i, batch in enumerate(_iterator):
        if i == cfg.TEST.SUBSET:
            break
        images, targets, image_ids, *_ = batch

        all_output = []
        mdetr_style_output = []
        with torch.no_grad():
            if cfg.TEST.USE_MULTISCALE:
                query_time = len(all_queries)
                for query_i in range(query_time):
                    if task == "detection":
                        captions = [all_queries[query_i] for ii in range(len(targets))]
                        positive_map_label_to_token = all_positive_map_label_to_token[query_i]
                    else:
                        captions = None
                        positive_map_label_to_token = None

                output = im_detect_bbox_aug(model, images, device, captions, positive_map_label_to_token)
                output = [o.to(cpu_device) for o in output]
                all_output.append(output)
            else:
                images = images.to(device)
                query_time = len(all_queries)

                for query_i in range(query_time):
                    if not isinstance(targets[0], dict): # For LVIS dataset and datasets directly copied from MDETR
                        targets = [target.to(device) for target in targets]
                    '''
                    different datasets seem to have different data format... For LVIS dataset, the target is a dictionary, while for modulatedDataset such as COCO/Flickr, the target is a BoxList
                    '''

                    if task == "detection":
                        captions = [all_queries[query_i] for ii in range(len(targets))]
                        positive_map_label_to_token = all_positive_map_label_to_token[query_i]
                    elif task == "grounding":
                        captions = [t.get_field("caption") for t in targets]
                        positive_map_eval = [t.get_field("positive_map_eval") for t in targets]
                        if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                            plus = 1
                        else:
                            plus = 0
                        assert(len(positive_map_eval) == 1) # Let's just use one image per batch
                        positive_map_eval = positive_map_eval[0]
                        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map_eval, plus=plus)
                    output = model(images, captions=captions, positive_map=positive_map_label_to_token)
                    output = [o.to(cpu_device) for o in output]

                    if "flickr" in cfg.DATASETS.TEST[0]:
                        output = output[0]
                        new_output = flickr_post_process(
                            output,
                            targets,
                            positive_map_label_to_token,
                            plus # This is only used in Flickr
                        )
                        mdetr_style_output.append(new_output)
                    elif "lvis" in cfg.DATASETS.TEST[0]:
                        output = output[0]
                        output = resize_box(output, targets)
                        scores = output.extra_fields["scores"]
                        labels = output.extra_fields["labels"]
                        boxes = output.bbox
                        mdetr_style_output.append((targets[0]["image_id"].item(), {"scores": scores, "labels": labels, "boxes": boxes}))
                    else:
                        all_output.append(output)
        if visualizer is not None:
            assert(len(all_output) == 1)
            if "lvis" in cfg.DATASETS.TEST[0]:
                scores = [o[1]["scores"] for o in mdetr_style_output]
                labels = [o[1]["labels"] for o in mdetr_style_output]
                boxes = [o[1]["boxes"] for o in mdetr_style_output]
                scores = torch.cat(scores, dim=0)
                labels = torch.cat(labels, dim=0)
                boxes = torch.cat(boxes, dim=0)
                visualizer_input = BoxList(boxes, output.size)
                visualizer_input.add_field("scores", scores)
                visualizer_input.add_field("labels", labels)
            else:
                visualizer_input = all_output[0][0] # single image_visualize

            image_id = dataset.ids[i]
            try:
                image_path = os.path.join(dataset.root, dataset.coco.loadImgs(image_id)[0]["file_name"])
                categories = dataset.coco.dataset["categories"]
            except:
                lvis = dataset.lvis
                img_id = dataset.ids[i]
                ann_ids = lvis.get_ann_ids(img_ids=img_id)
                target = lvis.load_anns(ann_ids)

                image_path = "DATASET/coco/" +  "/".join(dataset.lvis.load_imgs(img_id)[0]["coco_url"].split("/")[-2:])
                categories = dataset.lvis.dataset["categories"]

            image = load(image_path)
            no_background = True
            label_list = []
            for index, i in enumerate(categories):
                if not no_background or (i["name"] != "__background__" and i['id'] != 0):
                    label_list.append(i["name"])
            visualizer.entities =  label_list
            
            result, _ = visualizer.visualize_with_predictions(
                image,
                visualizer_input, 
                threshold,
                alpha=alpha,
                box_pixel=box_pixel,
                text_size=text_size,
                text_pixel=text_pixel,
                text_offset=text_offset,
                text_offset_original=text_offset_original,
                color=color,
            )
            imshow(result, "./visualize/img_{}.jpg".format(i))
        
        if evaluator is not None:
            evaluator.update(mdetr_style_output)
        else:
            output = [[row[_i] for row in all_output] for _i in range(len(all_output[0]))]
            for index, i in enumerate(output):
                output[index] = i[0].concate_box_list(i)

            results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})

    if evaluator is not None:
        evaluator.synchronize_between_processes()
        try:
            evaluator.accumulate()
        except:
            print("Evaluator has no accumulation, skipped...")
        score = evaluator.summarize()
        print(score)
        import maskrcnn_benchmark.utils.mdetr_dist as dist
        if is_main_process():
            if "flickr" in cfg.DATASETS.TEST[0]:
                write_flickr_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
            elif "lvis" in cfg.DATASETS.TEST[0]:
                write_lvis_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
        try:
            torch.distributed.barrier()
        except:
            print("Default process group is not initialized")
        return

    if evaluator is not None:
        predictions = mdetr_style_output
    else:
        predictions = results_dict
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    print("Accumulated results")
    if not is_main_process():
        return None

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args)
