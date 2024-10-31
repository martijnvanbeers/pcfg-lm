from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig

class PcfgConfig(PretrainedConfig):
    pass

class PcfgXlstmConfig(PcfgConfig):
      model_type = "xlstm"

      def __init__(self, vocab_size=-1, num_blocks=2, context_length=40, embedding_dim=64, pad_token_id=0, **kwargs):
          super().__init__(pad_token_id=pad_token_id, **kwargs)

          self.vocab_size = vocab_size
          self.num_blocks = num_blocks
          self.embedding_dim = embedding_dim
          self.mlstm_block = kwargs.pop("mlstm_block", {"mlstm": {"num_heads": 4}})
          self.slstm_block = kwargs.pop("slstm_block", {"slstm": {"num_heads": 4}})
          self.slstm_at = kwargs.pop("slstm_at", [ 1 ])
          self.context_length = context_length


#PcfgXlstmConfig.register_for_auto_class()
# register this way so save model doesn't create custom code
AutoConfig.register("xlstm", PcfgXlstmConfig)

