import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _C


# TODO: Use JIT to replace CUDA implementation in the future.
class _SigmoidFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = _C.sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(
            logits, targets, d_loss, num_classes, gamma, alpha
        )
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr


def token_sigmoid_softmax_focal_loss(pred_logits, targets, alpha, gamma, text_mask=None):
    # Another modification is that because we use the cross entropy version, there is no frequent or not frequent class.
    # So we temporarily retired the design of alpha.

    assert (targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to

    # reprocess target to become probability map ready for softmax
    targets = targets.float()
    target_num = targets.sum(-1) + 1e-8  # numerical stability
    targets = targets / target_num.unsqueeze(-1)  # T(x)

    if text_mask is not None:
        # reserve the last token for non object
        assert (text_mask.dim() == 2)
        text_mask[:, -1] = 1
        text_mask = (text_mask > 0).unsqueeze(1).repeat(1, pred_logits.size(1), 1)  # copy along the image channel
        pred_logits = pred_logits.masked_fill(~text_mask, -1000000)  # softmax

    out_prob = pred_logits.softmax(-1)

    filled_targets = targets.clone()
    filled_targets[filled_targets == 0] = 1.0

    weight = torch.clamp(targets - out_prob, min=0.001) / filled_targets
    weight = torch.pow(weight, gamma)  # weight = torch.pow(torch.clamp(target - out_prob, min=0.01), gamma)

    loss_ce = - targets * weight * pred_logits.log_softmax(
        -1)  # only those positives with positive target_sim will have losses.
    return loss_ce


def token_sigmoid_binary_focal_loss_v2(pred_logits, targets, alpha, gamma, text_mask=None):
    assert (targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to

    if text_mask is not None:
        assert (text_mask.dim() == 2)

    # We convert everything into binary
    out_prob = pred_logits.sigmoid()
    out_prob_neg_pos = torch.stack([1 - out_prob, out_prob], dim=-1) + 1e-8  # batch x boxes x 256 x 2
    weight = torch.pow(-out_prob_neg_pos + 1.0, gamma)

    focal_zero = - weight[:, :, :, 0] * torch.log(out_prob_neg_pos[:, :, :, 0]) * (
            1 - alpha)  # negative class
    focal_one = - weight[:, :, :, 1] * torch.log(out_prob_neg_pos[:, :, :, 1]) * alpha  # positive class
    focal = torch.stack([focal_zero, focal_one], dim=-1)
    loss_ce = torch.gather(focal, index=targets.long().unsqueeze(-1), dim=-1)
    return loss_ce


def token_sigmoid_binary_focal_loss(pred_logits, targets, alpha, gamma, text_mask=None):
    # binary version of focal loss
    # copied from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor with the reduction option applied.
    """
    assert (targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to

    bs, n, _ = pred_logits.shape
    if text_mask is not None:
        assert (text_mask.dim() == 2)
        text_mask = (text_mask > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, pred_logits.size(1), 1)  # copy along the image channel dimension
        pred_logits = torch.masked_select(pred_logits, text_mask)
        targets = torch.masked_select(targets, text_mask)

        # print(pred_logits.shape)
        # print(targets.shape)

    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


class TokenSigmoidFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(TokenSigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, text_masks=None, version="binary", **kwargs):
        if version == "binary":
            loss_func = token_sigmoid_binary_focal_loss
        elif version == "softmax":
            loss_func = token_sigmoid_softmax_focal_loss
        elif version == "binaryv2":
            loss_func = token_sigmoid_binary_focal_loss_v2
        else:
            raise NotImplementedError
        loss = loss_func(logits, targets, self.alpha, self.gamma, text_masks, **kwargs)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
