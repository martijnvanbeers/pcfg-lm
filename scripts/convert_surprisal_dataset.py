# coding: utf-8
import sys
import pathlib
from tqdm import tqdm
import pyarrow
import pyarrow.parquet
import pandas

myschema = pyarrow.schema([
        ('sentence', pyarrow.int64()),
        ('token', pyarrow.string()),
        ('surprisal', pyarrow.float32())
    ])

n = 0
sent = []
source_data = pathlib.Path(sys.argv[1])
path = source_data.parent
dest_data = path  / f"surprisal_{source_data.stem}.parquet"
with (
        source_data.open("r") as fh,
        pyarrow.parquet.ParquetWriter(dest_data, myschema) as pw
):
    for line in tqdm(fh):
        line = line.strip()
        if line[0] == "#":
            if line[1:] == "! Done":
                n += 1
                sentdf = pandas.DataFrame(sent)
                sent = []
                tb = pyarrow.lib.Table.from_pandas(sentdf, schema=myschema)
                pw.write_table(tb)

        else:
            token, score = line.split()
            sent.append({'sentence': n, 'token': token, 'surprisal': float(score)})
