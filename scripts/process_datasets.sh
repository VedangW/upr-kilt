BASE_DIR="/data/local/vw120/kilt_bm25"

python scripts/ml1_datasetprep.py \
    --in_file ${BASE_DIR}/datasets/fever-dev-kilt.jsonl \
    --out_file ${BASE_DIR}/processed_datasets/fever-dev-kilt-processed.json \
    --verbose