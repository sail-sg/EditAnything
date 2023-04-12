from .od_eval import do_od_evaluation


def od_to_grounding_evaluation(
        dataset,
        predictions,
        output_folder,
        box_only=False,
        iou_types=("bbox",),
        expected_results=(),
        expected_results_sigma_tol=4, ):
    return do_od_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
