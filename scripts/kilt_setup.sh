BASE_DIR="/data/local/vw120/kilt_bm25"

python kilt/prepare_fs.py --base_dir ${BASE_DIR}
python kilt/download_all_kilt_data.py --base_dir ${BASE_DIR}
python kilt/get_triviaqa_input.py --base_dir ${BASE_DIR}