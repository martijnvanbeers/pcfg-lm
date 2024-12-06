# coding: utf-8

import click
import pathlib

import logging

import json
import numpy
import torch

import datasets

from codecarbon import OfflineEmissionsTracker

from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        DataCollatorForLanguageModeling,
        TrainingArguments,
    )

from trainer import XlstmCapableTrainer

# to register the classes for transformers' Auto*
from amsterdamnlp.pcfgrammar import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#transformers.trainer.logger.setLevel(logging.DEBUG)

@click.command()
@click.pass_context
@click.argument("config_file", type=click.Path(path_type=pathlib.Path, exists=True))

def run(ctx, config_file: pathlib.Path):
    """
    Build a model according to the passed config
    """

    with config_file.open("r") as fh:
        config = json.load(fh)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if config['dataset'].startswith(config['resource_path']):
        dsd = datasets.DatasetDict.load_from_disk(config['dataset'])
    else:
        dsd = datasets.load_dataset(config['dataset'])

    model_config = AutoConfig.for_model(vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id, **(config['model']))
    model = AutoModelForCausalLM.from_config(model_config)


    train_args=TrainingArguments(
            **config['trainer'],
            fp16=torch.cuda.is_available(),
            group_by_length=True,
            auto_find_batch_size=False,
            do_eval=True,
        )

    trainer = XlstmCapableTrainer(
                model=model,
                tokenizer=tokenizer,
                args=train_args,
                data_collator=collator,
                train_dataset=dsd["train"],
                eval_dataset=dsd["valid"],
            )

    with OfflineEmissionsTracker(
            experiment_name=model_config.model_type,
            experiment_id=train_args.output_dir[28:],
            country_iso_code="NLD")
    as tracker:

        train_result = trainer.train()
    trainer.state.save_to_json(pathlib.Path(config['trainer']['logging_dir']) / "train_state.json")
    trainer._save_checkpoint(trainer.model, None)

if __name__ == '__main__':
    run()
