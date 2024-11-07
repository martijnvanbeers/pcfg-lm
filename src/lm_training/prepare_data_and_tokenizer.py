# coding: utf-8
import click
import pathlib
import datasets
from amsterdamnlp.pcfgrammar.tokenization_pcfg_fast import create_tokenizer_fast
from data import load_data

@click.command()
@click.pass_context
@click.argument("postfix", type=str)
@click.argument("dataset-dir", type=click.Path(path_type=pathlib.Path, exists=True))
@click.argument("save-dir", type=click.Path(path_type=pathlib.Path))
@click.option("--push-to-hub", is_flag=True, show_default=True, default=False, help="upload the tokenizer and dataset to the huggingface hub")
@click.option("--hub-org", type=str, help="huggingface hub organisation to use (This will set push_to_hub to True)")
@click.option("--private", is_flag=True, show_default=True, default=True, help="Set hub repositories to private on creation")
def main(ctx, postfix: str, dataset_dir: pathlib.Path, save_dir: pathlib.Path, push_to_hub: bool, hub_org: str):
    """
    Build a huggingface datasets.DatasetDict from text files and train a tokenizer on the train data.

    Optionally uploads the results to the huggingface hub.
    """
    if hub_org is not None:
        push_to_hub=True
    print("loading dataset")
    ds = datasets.load_dataset(str(dataset_dir))
    print("creating tokenizer")
    tokenizer = create_tokenizer_fast(postfix, ds['train']['text'], save_directory=save_dir / "tokenizers")
    repo = f"pcfg-tokenizer-{postfix}"
    if hub_org is not None:
        repo = f"hub_org/{repo}"
    #tokenizer.push_to_hub(repo, private=True)
    print("load and prepare data")
    dsd = load_data(tokenizer, dataset_dir)
    print("save data")
    dsd.save_to_disk(save_dir / "datasets" / postfix)
    repo = f"pcfg-dataset-{postfix}"
    if hub_org is not None:
        repo = f"hub_org/{repo}"
    #dsd.push_to_hub(repo)

if __name__ == '__main__':
    main()
