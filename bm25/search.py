import json
import argparse

from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher
from utils import read_json, write_json, read_config

def has_answer(hit, labels):
    present = False
    ### Implement
    
    return "true" if present else "false"

def retrieve(queries, num_candidates, searcher, pid2title):

    def get_text(hit, title):
        text = json.loads(hit.raw)['contents'][len(title):].strip()
        return text

    results = []
    for query in tqdm(queries):
        query_id = query['id']
        question = query['question']
        answers = query['answers']
        labels = query['labels']
        wiki_ids = set()
        for label in labels:
            if len(label):
                for prov_item in label['provenance']:
                    wiki_ids.add(str(prov_item['wikipedia_id']))

        hits = searcher.search(question, k=num_candidates)

        titles = [pid2title[hit.docid] for hit in hits]
        candidates = [
            {
                'id': hit.docid, 
                'wikipedia_id': json.loads(hit.raw)['wikipedia_id'], 
                'paragraph_id': json.loads(hit.raw)['passage_id'], 
                'start_span': json.loads(hit.raw)['start_span'], 
                'end_span': json.loads(hit.raw)['end_span'],
                'title': title,
                'text': get_text(hit, title), 
                'score': hit.score,
                'has_answer': "true" if str(json.loads(hit.raw)['wikipedia_id']) in wiki_ids else "false"
            }
        for hit, title in zip(hits, titles)]

        results.append({
            'id': query_id,
            'question': question,
            'answers': answers,
            'ctxs': candidates
        })

    return results


def read_queries(query_file, trunc=None, verbose=False):
    queries = read_json(query_file)

    if type(queries) == dict and 'content' in list(queries.keys()):
        queries = queries['content']

    if trunc:
        if verbose:
            print(f"Test run, truncating to {trunc} queries.")
        queries = queries[:trunc]

    return queries


def answer_queries(args):
    cfg = read_config(args.config_path)
    queries = read_queries(args.query_file, trunc=args.trunc, verbose=args.verbose)
    pid2title = read_json(cfg['title_path'])
    searcher = LuceneSearcher(str(cfg['index_dir']))

    results = retrieve(queries, 
                       args.num_cands,
                       searcher, 
                       pid2title)

    write_json(args.out_file, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trunc', type=int, 
                        default=-1, help='For testing. Only check first `trunc` queries.')
    parser.add_argument('--num_cands', type=int, 
                        required=True, help='Value of k (num. of candidates to retrieve).')
    parser.add_argument('--config_path', type=str, 
                        required=True, help='Path to config file')
    parser.add_argument('--query_file', type=str, 
                        required=True, help='Path to query file')
    parser.add_argument('--out_file', type=str, required=True, 
                        help='Path of output file.')
    parser.add_argument('--verbose', action='store_true', 
                        help='More detailed print statements.')
    args = parser.parse_args()

    args.trunc = args.trunc if args.trunc > 0 else None

    answer_queries(args)

if __name__ == '__main__':
    main()
