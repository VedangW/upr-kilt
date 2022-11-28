import json
import pathlib


def read_tsv(path, has_header=True):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [[x.strip() for x in row.split('\t')] for row in lines]

    if has_header:
        lines.pop(0)

    return lines


def omit_ends(string, omit_char):
    if len(string) < 2:
        return string
    return string[1:-1] if string[0] == omit_char and string[-1] == omit_char \
        else string

def read_json(path):
    """ Reads JSON file into dict. """

    with open(path, 'r') as f:
        data = json.load(f)

    return data


def write_json(path, data):
    """ Writes dict or list to JSON file. """

    with open(path, 'w') as f:
        json.dump(data)


def read_config(path):
    """ Reads config from a file. """

    cfg = read_json(path)

    cfg['base_dir'] = pathlib.Path(cfg['base_dir'])
    cfg['collection_dir'] = cfg['base_dir'] / cfg['collection_dir']
    cfg['title_path'] = cfg['base_dir'] / cfg['title_path']
    cfg['index_dir'] = cfg['base_dir'] / cfg['index_dir']

    return cfg


def init_file_structure(cfg):
    """
    Initializes the file structure required for the
    BM25 index to be created.
    """

    if not cfg['base_dir'].is_dir():
        raise ValueError(f"base_dir: '{cfg['base_dir']} is not a directory.")

    if not cfg['index_dir'].is_dir():
        cfg['index_dir'].mkdir()

    if not cfg['collection_dir'].is_dir():
        cfg['collection_dir'].mkdir()


def check_file_structure(cfg):
    """ 
    Checks if the file structure is correctly created and
    if the index has been created already.
    """

    is_dir = lambda x: x.exists() and x.is_dir()
    is_empty = lambda x: len(list(x.iterdir())) > 0

    fs_check = is_dir(cfg['base_dir']) and \
               is_dir(cfg['collection_dir']) and \
               is_dir(cfg['index_dir'])

    index_created = cfg['title_path'].is_file() and \
                    not is_empty(cfg['collection_dir']) and \
                    not is_empty(cfg['index_dir'])

    return fs_check, index_created


def main():
    cfg = read_config('bm25/config.json')
    init_file_structure(cfg)
    print(check_file_structure(cfg))

    print(read_tsv('bm25/sample.tsv'))


if __name__ == '__main__':
    main()
