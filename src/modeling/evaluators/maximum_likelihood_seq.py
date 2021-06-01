import collections

from .abstract import *
from torch.nn import functional as F

class MaximumLikelihoodSeq(Evaluator):
    text_key='question'
    def forward(self, model: nn.Module, batch: Any, **kwargs: Any) -> Dict[str, Tensor]:
        assert isinstance(batch, (dict, collections.OrderedDict, collections.UserDict,))

        # get input_ids
        assert f"{self.text_key}.input_ids" in batch.keys()
        input_ids = batch[f"{self.text_key}.input_ids"]
        assert isinstance(input_ids, torch.LongTensor)

        # get masking
        attention_mask = batch.get(f"{self.text_key}.attention_mask", None)

        # compute forward pass and loss
        logits = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits.permute(0, 2, 1), input_ids, reduce='none')

        if attention_mask is not None:
            loss = (loss * attention_mask).sum()/ attention_mask.sum()
            nll = (loss * attention_mask).sum(1).mean()
        else:
            loss = loss.mean()
            nll = loss.sum(1).mean()

        return {'loss': loss, 'nll': nll}