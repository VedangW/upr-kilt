BASE_DIR="/data/local/vw120/kilt_bm25"

if [ ! -d "${BASE_DIR}/bm25_outputs" ]; then
    mkdir "${BASE_DIR}/bm25_outputs"
fi

python bm25/search.py \
    --config_path bm25/config.json \
    --query_file ${BASE_DIR}/processed_datasets/fever-dev-kilt-processed.json \
    --out_file ${BASE_DIR}/bm25_results/retr.json \
    --num_cands 2 \
    --trunc 5
