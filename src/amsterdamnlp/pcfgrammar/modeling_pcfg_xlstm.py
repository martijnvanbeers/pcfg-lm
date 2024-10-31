from typing import Optional, Union
from dataclasses import dataclass
import dacite

import torch
from torch.nn import CrossEntropyLoss

from transformers import (
        AutoConfig,
        PreTrainedTokenizer,
        AutoModelForMaskedLM,
        AutoModelForCausalLM,
        AutoModel,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
import transformers.trainer
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

from .configuration_pcfg import PcfgXlstmConfig
@dataclass
class PcfgXlstmLMModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class PcfgXlstmLMHeadModel(PreTrainedModel):
    """
    tiny wrapper around xLSTMLMModel to make it compatible enough with
    transformers models to work in the Trainer
    """

    config_class = PcfgXlstmConfig
    def __init__(self, config):
        super().__init__(config)
        xlstm_config = dacite.from_dict(xLSTMLMModelConfig, config.to_dict())
        self.xlstm = xLSTMLMModel(xlstm_config)

    def forward(self, input_ids=None, labels=None, attention_mask=None):
        loss = None

        lm_logits = self.xlstm.forward(input_ids)
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return PcfgXlstmLMModelOutput(
                loss=loss,
                logits=lm_logits,
            )

# register our class with AutoModelForCausalLM
#PcfgXlstmLMHeadModel.register_for_auto_class(auto_class=AutoModelForCausalLM)

# register this way so save model doesn't create custom code
AutoModelForCausalLM.register(PcfgXlstmConfig, PcfgXlstmLMHeadModel)
