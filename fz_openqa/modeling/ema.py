import torch


class EMA(torch.nn.Module):
    def __init__(self, mu, model):
        super(EMA, self).__init__()
        self.mu = mu
        for name, param in model.named_parameters():
            if param.requires_grad:
                name = name.replace(".", "_")
                self.register_buffer(name, param.data.clone())

    def _update_param(self, name, x):
        name = name.replace(".", "_")
        stored_value = self._buffers[name]
        new_average = (1 - self.mu) * x + self.mu * stored_value
        self._buffers[name] = new_average.clone()
        return new_average

    @torch.no_grad()
    def forward(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self._update_param(name, param.data)
