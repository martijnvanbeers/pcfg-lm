from typing import List
import argparse

from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM
from minicons import scorer


def compute_clm_scores(eval_corpus_path: str, model_name: str, flatten=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ilm_model = scorer.IncrementalLMScorer(model_name, device)

    with open(eval_corpus_path) as f:
        eval_corpus = f.read().strip().split('\n')
        
    lm_scores = ilm_model.sequence_score(eval_corpus, reduction = lambda x: x)

    if flatten:
        lm_scores = torch.cat(lm_scores)

    return lm_scores


def parse_pcfg_scores(path: str, flatten=False):
    all_pcfg_scores = []
    
    with open(f'{path}.surprisal') as f:
        raw_scores = f.read().strip().split('\n')
        
    sentence_scores = []
    for line in raw_scores:
        if not line.startswith('#'):
            prob = -float(line.split()[1])
            sentence_scores.append(prob)
        elif len(sentence_scores) > 0:
            sentence_scores = torch.tensor(sentence_scores)
            all_pcfg_scores.append(sentence_scores)
            sentence_scores = []
            
    if flatten:
        all_pcfg_scores = torch.cat(all_pcfg_scores)
            
    return all_pcfg_scores


def eval_lm_to_pcfg(lm_scores: torch.Tensor, pcfg_scores: torch.Tensor):
    assert len(lm_scores) == len(pcfg_scores), f'{len(lm_scores)} != {len(pcfg_scores)}'
    
    rho = spearmanr(lm_scores, pcfg_scores)
    
    print(rho)
    
    pmin = min(min(lm_scores), min(pcfg_scores))
    pmax = max(max(lm_scores), max(pcfg_scores))    
    
    plt.figure(figsize=(5,5))
    plt.scatter(lm_scores, pcfg_scores)
    plt.xlim(pmin, pmax)
    plt.ylim(pmin, pmax)
    plt.xlabel(r"log P$_{LM}(w)$")
    plt.ylabel(r"log P$_{PCFG}(w)$")
    plt.plot([pmin, pmax], [pmin, pmax], '--', color='0.5', lw=1)
    plt.title(fr"$\rho$: {rho[0]:.3f}")
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", help="Path to evaluation corpus", required=True)
    parser.add_argument("-m", "--model_name", help="Path saved LM", required=True)
    parser.add_argument("-p", "--pcfg_scores", help="Path saved extracted PCFG scores", required=True)

    args = vars(parser.parse_args())

    lm_scores = compute_clm_scores(args['corpus'],  args['model_name'], flatten=True)
    pcfg_scores = parse_pcfg_scores(args['pcfg_scores'], flatten=True)

    eval_lm_to_pcfg(lm_scores, pcfg_scores)
