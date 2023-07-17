# upr-kilt
Unsupervised Passage Retrieval for Question Answering, Fact Checking, and Entity Linking on the KILT benchmark using the T5 language model series.

## Setup

Choose a base directory, `BASE_DIR`, which will be large enough to hold upto 60 GB of data.

#### Clone

```shell
# Clone this repo
git clone git@github.com:VedangW/upr-kilt.git
```

#### Install requirements

```shell
# Create a conda environment
conda create -n uprkilt python=3.8
conda activate uprkilt

# Install KILT
git clone git@github.com:facebookresearch/KILT.git
cd KILT
python setup.py install
cd ..

# Install other packages
cd upr-kilt
pip install -r requirements.txt
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
conda install -c pytorch faiss-gpu
```

#### Download KILT datasets

```shell
export BASE_DIR=... # Add base dir here

# Change BASE_DIR in kilt_setup.sh
bash scripts/kilt_setup.sh

# Download knowledge source
mkdir ${BASE_DIR}/knowledge_base
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json -o ${BASE_DIR}/knowledge_base/kilt_knowledgesource.json

# Preprocess datasets
bash scripts/process_datasets.sh
```

#### Build Lucene index for fast retrieval from BM25

Create a JSON file in `upr-kilt/bm25` called `config.json`. This file should have the following structure:

```json
{
  "base_dir": "/path/to/BASE_DIR",
  "collection_dir": "collection",
  "title_path": "pid2title.json",
  "index_dir": "index"
}
```

Here, `collection_dir`, `title_path`, and `index_dir` can be anything you want. These are the directories used to store the sharded knowledge source, a dictionary mapping passage ID to passage title, and the directory containing the index respectively.

```shell
# Build index
bash scripts/bm25_build_index.sh

# Create supporting directories
mkdir ${BASE_DIR}/bm25_outputs # stores search outputs of BM25
mkdir ${BASE_DIR}/cache        # cache dir
mkdir ${BASE_DIR}/reranked     # stores reranked output
```

At the end of the setup process, the structure of the base directory should be as follows:

```
BASE_DIR
  |
  |--- bm25_outputs
  |--- cache
  |--- collection
  |--- pid2title.json
  |--- index
  |--- knowledge_base
  |--- datasets
  |--- processed_datasets
  |--- reranked
```

## Usage

#### Search using BM25

```shell
# BM25 search
bash scripts/bm25_search.sh
```

#### Re-rank documents

Change `DATASET` in `rerank.sh` as you run this file. `DATA_DIR` is any directory where you store large data. For me, `BASE_DIR` is inside `DATA_DIR`.

```shell
# Rerank documents
export DATA_DIR=...
bash scripts/rerank.sh
```
