from typing import Optional, Iterator, Tuple

import json
import pathlib

import tokenizers
from transformers import (
        PreTrainedTokenizer,
        AutoTokenizer,
        PretrainedConfig,
    )

class PcfgTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "vocab.json",
    }
    def __init__(self, vocab_file:str, **kwargs):

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, sen: str):
        # strip() because the the Trie in PretrainedTokenizer
        # leaves the spaces around special tokens in the sentence
        # pieces we get as input here:
        # "A <mask> sentence" ends up as "A " and " sentence" in `sen`
        return sen.strip().split(" ")

    def _convert_token_to_id(self, token: str):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory:str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        dp = pathlib.Path(save_directory)
        if filename_prefix is not None:
            fp = dp / f"{prefix}{self.vocab_files_names['vocab_file']}"
        else:
            fp = dp / self.vocab_files_names['vocab_file']
        with fp.open("w") as fh:
            json.dump(self.get_vocab(), fh)

        return (str(fp),)
