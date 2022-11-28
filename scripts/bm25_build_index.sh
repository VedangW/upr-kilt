PREFIX="/data/local/vw120/upr/downloads/data/"

python bm25/build_index.py \
    --config_path bm25/config.json \
    --passages ${PREFIX}/wikipedia-split/psgs_toy.tsv \
    --n_threads 1
