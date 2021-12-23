#!/usr/bin/env python3
'''
Join the Pontiki SemEval datasets to get a wider-domain point of comparison set.

join_datasets.py in GTS
2/22/21 Copyright (c) Gilles Jacobs
'''
from pathlib import Path
import json
import random

random.seed(421992)

dataset_dir = Path("../data")
semeval = ["lap14", "res14", "res15", "res16"]
splits = ["train", "dev", "test"]
data_splits = {}
out_dir = dataset_dir / "joinedsemeval"
out_dir.mkdir(parents=True, exist_ok=True)
for s in splits:
    joined = []
    for d in semeval:
        fp = dataset_dir / d / f'{s}.json'
        with open(fp, "rt") as split_in:
            data = json.load(split_in)
            for inst in data:
                new_id = f'{d}_{inst["id"]}'
                for tpl in inst['triples']:
                    tpl['uid'] = new_id + '-' + tpl['uid'].split('-')[-1]
                inst['id'] = new_id
            joined.extend(data)
    random.shuffle(joined)
    with open(out_dir / f'{s}.json', 'wt') as split_out:
        json.dump(joined, split_out)
