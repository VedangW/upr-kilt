import sys
import json

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
        j_obj = change(obj)
        final_obj['content'].append(j_obj)
    file2.write(json.dumps(final_obj))
    file2.close()