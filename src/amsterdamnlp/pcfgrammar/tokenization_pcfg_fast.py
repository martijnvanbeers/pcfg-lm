from typing import Optional, Iterator, Tuple

import json
import pathlib

import tokenizers
from transformers import (
        PreTrainedTokenizerFast,
        PreTrainedTokenizer,
        AutoTokenizer,
        PretrainedConfig,
    )

from .configuration_pcfg import PcfgConfig
from .tokenization_pcfg import PcfgTokenizer

class PcfgTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = PcfgTokenizer
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

# register our class with AutoTokenizer
AutoTokenizer.register(PcfgConfig, slow_tokenizer_class=PcfgTokenizer, fast_tokenizer_class=PcfgTokenizerFast)

def create_tokenizer_fast(name_postfix: str, training_data: Iterator, save_directory: Optional[str] = None, name="pcfg-tokenizer") -> PcfgTokenizerFast:
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()

    trainer = tokenizers.trainers.WordLevelTrainer(special_tokens=["<cls>", "<pad>", "<unk>", "<mask>"])
    tokenizer.train_from_iterator(training_data, trainer=trainer)
    bos_token_id = tokenizer.token_to_id("<cls>")
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
            single=f"<cls>:0 $A:0",
#            pair=f"<cls>:0 $A:0 <cls>:1 $B:1",
            special_tokens=[
                    ("<cls>", bos_token_id,)
                ],
        )

    wrapped_tokenizer = PcfgTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<cls>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )

    if save_directory is None:
        save_directory = "."
    save_directory = pathlib.Path(save_directory)
    wrapped_tokenizer.save_pretrained(str(save_directory / f"{name}-{name_postfix}"))
    # save the vocabulary too, so we can also use the non-fast tokenizer
    wrapped_tokenizer.save_vocabulary(str(save_directory / f"{name}-{name_postfix}"))
    return wrapped_tokenizer
