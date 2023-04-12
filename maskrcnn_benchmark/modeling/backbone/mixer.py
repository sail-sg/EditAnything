import torch
from torch import nn

class MixedOperationRandom(nn.Module):
    def __init__(self, search_ops):
        super(MixedOperationRandom, self).__init__()
        self.ops = nn.ModuleList(search_ops)
        self.num_ops = len(search_ops)

    def forward(self, x, x_path=None):
        if x_path is None:
            output = sum(op(x) for op in self.ops) / self.num_ops
        else:
            assert isinstance(x_path, (int, float)) and 0 <= x_path < self.num_ops or isinstance(x_path, torch.Tensor)
            if isinstance(x_path, (int, float)):
                x_path = int(x_path)
                assert 0 <= x_path < self.num_ops
                output = self.ops[x_path](x)
            elif isinstance(x_path, torch.Tensor):
                assert x_path.size(0) == x.size(0), 'batch_size should match length of y_idx'
                output = torch.cat([self.ops[int(x_path[i].item())](x.narrow(0, i, 1))
                                    for i in range(x.size(0))], dim=0)
        return output