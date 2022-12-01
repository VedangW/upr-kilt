import sys
import json
import pathlib
import argparse

def change(obj):
    new={}
    j_obj=json.loads(obj)
    new['id']=j_obj['id']
    new['question'] = j_obj['input']
    new['answers']=[]
    for objs in j_obj['output']:
        if objs.get('answer'):
            new['answers'].append(objs['answer'])
    if len(new['answers'])!=0:
        new['labels']=[]
    for p_set in j_obj['output']:
        if p_set.get('provenance'):
            if p_set['provenance'][0].get('section'):
                del p_set['provenance'][0]['section']
            if p_set['provenance'][0].get('meta'):
                del p_set['provenance'][0]['meta']
            new['labels'].append({"provenance":p_set.get('provenance')})
    return new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=pathlib.Path, 
                        required=True, help='JSON file to process.')
    parser.add_argument('--out_file', type=pathlib.Path, 
                        required=True, help='Output of preprocessing stored here.')
    parser.add_argument('--verbose', action='store_true', 
                        help='More detailed print statements.')
    args = parser.parse_args()

    if not args.in_file.exists():
        raise ValueError(f"Input file `{args.in_file}` doesn't exist!")
    
    if args.out_file.exists() and args.verbose:
        print(f"Out file `{args.out_file}` already exists. Overwriting.")

    final_obj = {}
    final_obj['content'] = []

    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    for obj in lines:
        obj = change(obj)
        final_obj['content'].append(obj)
    
    print(len(final_obj['content']))

    with open(args.out_file, 'w') as f:
        json.dump(final_obj, f)


if __name__ == '__main__':
    main()