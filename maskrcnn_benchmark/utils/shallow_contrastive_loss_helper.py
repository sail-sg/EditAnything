import torch
import maskrcnn_benchmark.utils.dist as dist


def normalized_positive_map(positive_map):
    positive_map = positive_map.float()
    positive_map_num_pos = positive_map.sum(2)
    positive_map_num_pos[positive_map_num_pos == 0] = 1e-6
    positive_map = positive_map / positive_map_num_pos.unsqueeze(-1)
    return positive_map


def pad_tensor_given_dim_length(tensor, dim, length, padding_value=0, batch_first=True):
    new_size = list(tensor.size()[:dim]) + [length] + list(tensor.size()[dim + 1:])
    out_tensor = tensor.data.new(*new_size).fill_(padding_value)
    if batch_first:
        out_tensor[:, :tensor.size(1), ...] = tensor
    else:
        out_tensor[:tensor.size(0), ...] = tensor
    return out_tensor


def pad_random_negative_tensor_given_length(positive_tensor, negative_padding_tensor, length=None):
    assert positive_tensor.shape[0] + negative_padding_tensor.shape[0] == length
    return torch.cat((positive_tensor, negative_padding_tensor), dim=0)


def gather_tensors(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_dist_avail_and_initialized():
        return torch.stack([tensor], dim=0)

    total = dist.get_world_size()
    rank = torch.distributed.get_rank()
    # gathered_normalized_img_emb = [torch.zeros_like(normalized_img_emb) for _ in range(total)]
    # torch.distributed.all_gather(gathered_normalized_img_emb, normalized_img_emb)

    tensors_gather = [
        torch.zeros_like(tensor)
        for _ in range(total)
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # need to do this to restore propagation of the gradients
    tensors_gather[rank] = tensor
    output = torch.stack(tensors_gather, dim=0)
    return output


def convert_to_roi_format(boxes):
    concat_boxes = boxes.bbox
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.full((len(boxes), 1), 0, dtype=dtype, device=device)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois