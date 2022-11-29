QUERY_PATH=/data/local/vw120/downloads/data/retriever/nq-dev.json

python bm25/search.py \
    --config_path bm25/config.json \
    --query_file ${QUERY_PATH} \
    --out_file retr.json \
    --num_cands 2 \
    --trunc 5