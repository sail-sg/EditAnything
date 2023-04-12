from ... import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .vg import vg_evaluation
from .box_aug import im_detect_bbox_aug
from .od_to_grounding import od_to_grounding_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))


def evaluate_mdetr(dataset, predictions, output_folder, cfg):
   
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
