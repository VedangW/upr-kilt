import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=pathlib.Path, 
                    required=True, help='All data would be stored at base_dir/datasets.')
parser.add_argument('--reset', action='store_true',
                    help='If directory already exists, remove it.')
args = parser.parse_args()

if not args.base_dir.exists():
    raise ValueError(f"The base path doesn't exist!")

datasets = args.base_dir / "datasets"
trivia_path = args.base_dir / "triviaqa-rc"
cache = args.base_dir / "cache"

def init_dir(path, args):
    if not path.exists():
        path.mkdir()
    elif path.exists() and args.reset:
        shutil.rmtree(path)
    else:
        raise ValueError(f"path: {path} already exists.")

init_dir(datasets)
init_dir(trivia_path)
init_dir(cache)
