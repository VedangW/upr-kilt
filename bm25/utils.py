import os
import csv
import json

def read_file(infile, handle_file, verbose=False, skip_first_line=False):
    if verbose:
        print(f'Opening "{infile}"...')
    data = None
    with open(infile) as f:
        if skip_first_line:
            f.readline()
        data = handle_file(f)
    if verbose:
        print('  Done.')
    return data


def read_tsv(path, row_fn=lambda x: x, verbose=False, skip_first_line=False):
    handler = lambda f: [row_fn(line.split('\t')) for line in f.readlines()]
    return read_file(path, handler, verbose=verbose,
                     skip_first_line=skip_first_line)


def omit_ends(string, omit_char):
    if len(string) < 2:
        return string
    return string[1:-1] if string[0] == omit_char and string[-1] == omit_char \
        else string

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data)

def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    return cfg

def init_file_structure(cfg):
    if not os.path.isdir(cfg.data_dir):
        raise ValueError(f"data_dir: '{cfg.data_dir}' doesn't exist.")

