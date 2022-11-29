COLLECTION_DIR=/data/local/vw120/kilt_bm25/collection_passages
INDEX_DIR=/data/local/vw120/kilt_bm25/index

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${COLLECTION_DIR} \
  --index ${INDEX_DIR} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw