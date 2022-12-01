import sys
import json


def process_json(obj):
    j_obj = json.loads(obj)
    j_obj['answers'] = []
    for objs in j_obj['output']:
        if objs.get('answer'):
            j_obj['answers'].append(objs['answer'])
            del objs['answer']
    j_obj['labels'] = j_obj['output']
    del j_obj['output']
    for p_set in j_obj['labels']:
        if p_set.get('provenance'):
            if p_set['provenance'][0].get('section'):
                del p_set['provenance'][0]['section']
            if p_set['provenance'][0].get('meta'):
                del p_set['provenance'][0]['meta']
    if j_obj.get('meta'):
        del j_obj['meta']
    return j_obj


if __name__ == '__main__':
    final_obj = {}
    final_obj['content'] = []
    input_file = ""
    out_file = ""
    if len(sys.argv) > 1:
        input_file = str(sys.argv[1])
        out_file = str(sys.argv[2])
    file1 = open(input_file, 'r')
    file2 = open(out_file, 'w')
    Lines = file1.readlines()
    file1.close()
    for obj in Lines:
        j_obj = process_json(obj)
        final_obj['content'].append(j_obj)
    file2.write(json.dumps(final_obj))
    file2.close()
