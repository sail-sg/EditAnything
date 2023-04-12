from copy import deepcopy
from collections import OrderedDict
import torch


class ModelEma:
    def __init__(self, model, decay=0.9999, device=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_is_dp = hasattr(self.ema, 'module')
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)

        assert isinstance(checkpoint, dict)
        if 'model_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_ema'].items():
                if self.ema_is_dp:
                    name = k if k.startswith('module') else 'module.' + k
                else:
                    name = k.replace('module.', '') if k.startswith('module') else k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)

    def state_dict(self):
        return self.ema.state_dict()

    def update(self, model):
        pre_module = hasattr(model, 'module') and not self.ema_is_dp
        with torch.no_grad():
            curr_msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                k = 'module.' + k if pre_module else k
                model_v = curr_msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)

