import os
import math
import argparse

from tqdm import tqdm

from jnius import autoclass
from upr_kilt.bm25.utils import read_config, init_file_structure, \
                  write_json, read_tsv
from upr_kilt.bm25.serializers import TsvSerializer


def read_passages(cfg):
    # TODO: Change paths

    serializer = TsvSerializer()
    documents = read_tsv(cfg['passages'], 
                         row_fn=serializer.serialize,
                         skip_first_line=True)

    return documents, serializer.pid2title


def build_index(cfg, args):
    # TODO: Implement sub-functions

    init_file_structure(cfg)

    documents, pid2title = read_passages(cfg, args)

    shard_template = os.path.join(cfg['collection_dir'], 'shard{}.json')
    shard_size = math.ceil(len(documents) / args.num_shards)
    for shard_id in (tqdm(range(args.num_shards)) if args.verbose else
                     range(args.num_shards)):
        shard = documents[shard_id * shard_size:(shard_id + 1) * shard_size]
        outpath = shard_template.format(shard_id)
        write_json(outpath, shard)
    write_json(cfg['title_path'], pid2title)

    args_external = [
        '-collection', 'JsonCollection',
        '-generator', 'DefaultLuceneDocumentGenerator',
        '-threads', str(args.n_threads),
        '-input', cfg['collection_dir'],
        '-index', cfg['index_dir'],
        '-storePositions',
        '-storeRaw'
    ]

    JIndexCollection = autoclass('io.anserini.index.IndexCollection')
    JIndexCollection.main(args_external)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        required=True, help='Path to config file.')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='No. of threads for indexing.')
    args = parser.parse_args()
    cfg = read_config(args.config_path)

    build_index(cfg, args)
        
    
if __name__ == '__main__':
    main()
