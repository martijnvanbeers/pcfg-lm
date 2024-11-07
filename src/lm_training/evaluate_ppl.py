# coding: utf-8
import click
import pathlib
import json

import datasets
from amsterdamnlp.pcfgrammar import *

@click.command()
@click.pass_context
@click.argument("config", type=click.Path(path_type=pathlib.Path, exists=True))
@click.argument("step", type=int)
def run(config: pathlib.Path, step):
    """
    Calculate perplexities on a model checkpoint
    """

    with config.open("r") as fh:
        config = json.load(fh)

    if config['dataset'].startswith(config['resource_path']):
        dsd = datasets.DatasetDict.load_from_disk(config['dataset'])
    else:
        dsd = datasets.load_dataset(config['dataset'])

    model = pathlib.Path(config['trainer']['output_dir']) / f"checkpoint-{step}"
    savedir = model

    perplexity = Perplexity()

    testds = dsd['test']

    print(f"Evaluating {str(model)}")
    results = (perplexity
               .compute(
                    model_id=str(model),
                    data=testds['text'],
                    add_start_token=True,
                    device="cuda",
                    max_length=40,
                    batch_size=1,
                )
            )

    print("-" * 24)
    print(" Model mean perplexity")
    print("                ", round(results["mean_perplexity"], 2))
    print("-" * 24)

    print("Saving results to", savedir / "ppl_eval.json")
    with (savedir / "ppl_eval.json").open("w") as fh:
        json.dump(results, fh)

if __name__ == '__main__':
    run()
