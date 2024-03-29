# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import requests
from tqdm.auto import tqdm

urls = [
    "http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/fever-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/nq-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/hotpotqa-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/hotpotqa-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/hotpotqa-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/triviaqa-train_id-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/triviaqa-dev_id-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/triviaqa-test_id_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/eli5-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/eli5-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/aidayago2-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/aidayago2-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/aidayago2-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/wned-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/wned-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/cweb-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/cweb-test_without_answers-kilt.jsonl",
]

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, 
                    required=True, help='All data would be stored at base_dir/datasets.')
args = parser.parse_args()

for url in urls:
    base = url.split("/")[-1]
    filename = f"{args.base_dir}/datasets/{base}"
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=base)
    with open(filename, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()