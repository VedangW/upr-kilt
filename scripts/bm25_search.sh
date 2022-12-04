BASE_DIR="/data/local/vw120/kilt_bm25"

if [ ! -d "${BASE_DIR}/bm25_outputs" ]; then
    echo "Output dir not created. Creating."
    mkdir "${BASE_DIR}/bm25_outputs"
fi

for dataset in aidayago2 cweb eli5 fever hotpotqa nq triviaqa wned
do
    echo "Dataset: ${dataset}."
    python bm25/search.py \
        --config_path bm25/config.json \
        --query_file ${BASE_DIR}/processed_datasets/${dataset}-dev-kilt-processed.json \
        --out_file ${BASE_DIR}/bm25_outputs/${dataset}-dev-bm25.json \
        --num_cands 1000 \
        --verbose
done