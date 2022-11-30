import os
import math
import pathlib
import argparse
import subprocess

from tqdm import tqdm

from jnius import autoclass
from utils import read_config, init_file_structure, \
                  write_json, read_tsv, check_file_structure
from serializers import UprSerializer


def read_passages(args):
    # TODO: Change paths

    serializer = UprSerializer()
    documents = read_tsv(args.passages,
                         serialize_fn=serializer.serialize,
                         has_header=True)

    return documents, serializer.pid2title


def build_index(args):
    """
    Builds index using Pyserini.
    """

    # Read config and initialize file structure
    cfg = read_config(args.config_path)
    init_file_structure(cfg, clear=args.clear_fs)
    is_correct, index_created = check_file_structure(cfg)

    # base_dir -> index, collection_passages

    assert is_correct, "F.S. not created correctly!"
    assert not index_created, "Index should be empty!"

    # Read passages from passages file
    documents, pid2title = read_passages(args)

    # Shard documents and save to file
    shard_template = os.path.join(cfg['collection_dir'], 'shard{}.json')
    shard_size = math.ceil(len(documents) / args.n_shards)
    for shard_id in tqdm(range(args.n_shards)):
        shard = documents[shard_id * shard_size:(shard_id + 1) * shard_size]
        outpath = shard_template.format(shard_id)
        write_json(outpath, shard)
    write_json(cfg['title_path'], pid2title)


    program = ['python', '-m', 'pyserini.index.lucene']

    # Create index
    args_external = [
        '--collection', 'JsonCollection',
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', str(args.n_threads),
        '--input', str(cfg['collection_dir']),
        '--index', str(cfg['index_dir']),
        '--storePositions',
        '--storeRaw'
    ]

    process = subprocess.Popen(
        program + args_external, 
        stdout=subprocess.PIPE
    )

    output, error = process.communicate()
    
    if error:
        print(f"Error: {error}")
    
    print(output.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=pathlib.Path,
                        required=True, help='Path to config file.')
    parser.add_argument('--passages', type=pathlib.Path,
                        required=True, help='Path to passages file.')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='No. of threads for indexing.')
    parser.add_argument('--n_shards', type=int, default=1,
                        help='No. of shards in which to store passages.')
    parser.add_argument('--clear_fs', action='store_true', 
                        help='Clear index dirs if they already exist.')
    args = parser.parse_args()

    build_index(args)

    
if __name__ == '__main__':
    main()
