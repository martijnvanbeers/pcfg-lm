from typing import List
from transformers import Trainer

class XlstmCapableTrainer(Trainer):
    def get_decay_parameters(self, model) -> List[str]:
        xlstm = getattr(model, xlstm, None)
        if xlstm is None:
            return super().get_decay_parameters(model)
        else:
            return xlstm.get_weight_decay_optim_group_param_names()[0]
