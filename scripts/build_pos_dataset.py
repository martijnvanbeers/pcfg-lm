# coding: utf-8
import sys
import pathlib
from collections import defaultdict
import re
from nltk import Tree
import pandas

treefile = pathlib.Path(sys.argv[1])
sents = []
with treefile.open("r") as fh:
    for n, line in enumerate(fh):
        line = line.strip()
        tree = Tree.fromstring(line)
        sent_df = pandas.DataFrame(tree.pos(), columns=["token", "pos"])
        sent_df['sent'] = n
        sents.append(sent_df)

complete = pandas.concat(sents)
complete.reset_index(names="token_ix")
complete['pos'] = complete['pos'].str.split("_").str[0]
complete.to_parquet(treefile.with_suffix(".parquet"))
