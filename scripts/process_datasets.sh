for dataset in aidayago2 cweb eli5 fever hotpotqa nq triviaqa wned
do
    infile="${BASE_DIR}/datasets/${dataset}-dev-kilt.jsonl"
    outfile="${BASE_DIR}/processed_datasets/${dataset}-dev-kilt-processed.json"

    python kilt/process_dataset.py \
        --in_file ${infile} \
        --out_file ${outfile} \
        --verbose
done