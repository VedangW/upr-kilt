export DATA_DIR="/freespace/local/vw120"
export TRANSFORMERS_CACHE="${DATA_DIR}/transformer_cache"

WORLD_SIZE=2

WORLD_SIZE=${WORLD_SIZE} python -m torch.distributed.launch \
    --nproc_per_node ${WORLD_SIZE} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    upr/upr_test.py \
    --dataset "/freespace/local/vw120/kilt_bm25/bm25_outputs/wned-dev.json"