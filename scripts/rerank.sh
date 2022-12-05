#! /bin/bash

export TRANSFORMERS_CACHE="${DATA_DIR}/transformer_cache"

BASE_DIR="${DATA_DIR}/kilt_bm25"
DATASET="hotpotqa"
SPLIT="dev"

MODEL="T0_3B"
HF_MODEL="bigscience/${MODEL}"
# MODEL="t5-v1_1-base"
# HF_MODEL="google/${MODEL}"
# Other possible options are MODEL="t5-v1_1-xl / t5-xl-lm-adapt and HF_MODEL="google/${MODEL}"

RETRIEVER="bm25"
TOPK=1000

if [[ "${DATA_DIR}" == "/data/vw120" ]]
then
  EVIDENCE_DATA_PATH="${BASE_DIR}/knowledge_base/passages_kb_complete.tsv"
else
  EVIDENCE_DATA_PATH="/data/local/gg676/KILT/knowledge_base_paragraphs/passages_kb_complete.tsv"
fi

echo "Evidence data at: ${EVIDENCE_DATA_PATH}"

WORLD_SIZE=4
DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${WORLD_SIZE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"

# VERBALIZER='"Please write a claim based on this passage."'

ARGS=" \
  --num-workers 2 \
  --log-interval 1 \
  --topk-passages ${TOPK} \
  --shard-size 32 \
  --hf-model-name ${HF_MODEL} \
  --use-gpu \
  --use-bf16 \
  --report-topk-accuracies 1 5 20 100 \
  --evidence-data-path ${EVIDENCE_DATA_PATH} \
  --retriever-topk-passages-path ${BASE_DIR}/bm25_outputs/${DATASET}-${SPLIT}.json \
  --reranker-output-dir ${BASE_DIR}/reranked/ \
  --merge-shards-and-save \
  --special-suffix ${DATASET}-${SPLIT}-plm-${MODEL}-topk-${TOPK}"

# `--use-bf16` option provides speed ups and memory savings on Ampere GPUs such as A100 or A6000.
# However, when working with V100 GPUs, this argument should be removed.


COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} upr/upr.py ${ARGS}"
eval "${COMMAND}"
exit
