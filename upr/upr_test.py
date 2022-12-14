from abc import ABC
import os
import json
import random
import pathlib
import argparse
import numpy as np
from torch.utils.data import Dataset
from utils import print_rank_0
from utils.initialize import initialize_distributed
from IPython import embed
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
import re

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=pathlib.Path, 
                    required=True, help='Path to dataset.')
parser.add_argument('--sort-by-score', action='store_true')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher.')
parser.add_argument('--main-port', type=int, default=29500,
                    help='Main port number.')
parser.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100, 1000],
                    help="Which top-k accuracies to report (e.g. '1 5 20')")

args = parser.parse_args()

args.rank = int(os.getenv('RANK', '0'))
args.world_size = int(os.getenv("WORLD_SIZE", '1'))
args.local_rank = int(os.environ['LOCAL_RANK'])

initialize_distributed(args)

if not args.dataset.exists():
    raise ValueError("Provided path doesn't exist.")


class OpenQADataset(ABC, Dataset):
    def __init__(self, task_name, dataset_name, filepath, sample_rate):
        self.task_name = task_name
        self.dataset_name = dataset_name
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        self.samples = self.load_dataset(filepath)

        if sample_rate < 1:  # subsample
            k = int(len(self.samples) * sample_rate)
            self.samples = random.sample(self.samples, k)

        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))

        if "trivia" in filepath or 'webq' in filepath or 'entity-questions' in filepath \
                or "BEIR" in filepath or "squad" in filepath:
            self.ques_punc = ""
        elif "nq" in filepath or "efficientqa" in filepath:
            self.ques_punc = "?"
        else:
            self.ques_punc = ""

    def extract_mention(self, ctxt, start='[START_ENT]', end='[END_ENT]'):
        """ Used for Entity Linking. """

        ctxt_left, right = ctxt.split(start)
        mention, ctxt_right = right.split(end)
        
        ctxt_left, mention, ctxt_right = \
            ctxt_left.strip(), mention.strip(), ctxt_right.strip()
        
        return ctxt_left + ' <extra_id_0> ' + ctxt_right, mention

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # These [CLS] and [SEP] tokens exist due to BERT tokenization, so we need to remove them
        if "[CLS]" and "[SEP]" in row['question']:
            row['question'] = " ".join(row['question'].split()[1:-1])

        if self.task_name == "reranking":
            decoder_prompt = "Question: {}{}".format(row['question'], self.ques_punc)
            masked_question = ""
        elif self.task_name == "fact_checking":
            decoder_prompt = "Claim: {}{}".format(row['question'], self.ques_punc)
            masked_question = ""
        elif self.task_name == "entity_linking":
            masked_question, decoder_prompt = self.extract_mention(row['question'])
        else:
            raise AssertionError("invalid --task-name argument {}".format(self.task_name))

        encoder_contexts = None
        if 'ctxs' in row:
            encoder_contexts = row['ctxs']
        elif 'contexts' in row:
            encoder_contexts = row['contexts']

        answers = row['answers']

        sample = {'id': idx,
                  'encoder_ids': encoder_contexts,
                  'decoder_ids': decoder_prompt,
                  'question': row['question'],
                  'masked_question': masked_question,
                  'answers': answers}
        return sample

    @staticmethod
    def load_dataset(filepath):
        with open(filepath) as fp:
            data = json.load(fp)

        # condition for interfacing with pyserineni BM25 outputs
        if isinstance(data, dict):
            return list(data.values())
        else:
            return data

dataset = OpenQADataset("reranking",
                        "open-domain retrieval",
                        str(args.dataset),
                        1.)

print(f"Question: {dataset[2332]['question']}")

gold_passage = "Robert Damien Bale Croft MBE (born 25 May 1970) is a former Welsh cricketer who played international cricket for England. He is an off-spin bowler who played for Glamorgan and captained the county from 2003 to 2006. He retired from first class cricket at the end of the 2012 season, having played county cricket for 23 seasons. He commentates on cricket occasionally for Sky Sports."

q = dataset[2332]['question']
context_left, mention = text.split('[START_ENT]')
mention, context_right = mention.split('[END_ENT]')

text = context_left + ' <extra_id_0> ' + context_right

# text = f"Passage: {q}. Please write a claim based on this passage."

print(text)

T5_PATH = 't5-base' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# Input text
# text = 'India is a <extra_id_0> of the world. </s>'

encoded = t5_tokenizer.encode_plus(text, 
                                   add_special_tokens=True,
                                   max_length=1024, 
                                   return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

# Generaing 20 sequences with maximum length set to 5
outputs = t5_mlm.generate(input_ids=input_ids, 
                          num_beams=200, num_return_sequences=20,
                          max_length=5)

_0_index = text.index('<extra_id_0>')
_result_prefix = text[:_0_index]
_result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>

def _filter(output, end_token='<extra_id_1>'):
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if end_token in _txt:
        _end_token_index = _txt.index(end_token)
        return _result_prefix + _txt[:_end_token_index] + _result_suffix
    else:
        return _result_prefix + _txt + _result_suffix

results = list(map(_filter, outputs))
results

def calculate_topk_hits(scores, max_k):
        top_k_hits = [0] * max_k
        for question_hits in scores:
            best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        return top_k_hits

def compute_topk_recall(answers_list, string_prefix='BM25'):
    topk_hits = calculate_topk_hits(answers_list, max_k=args.report_topk_accuracies[-1])

    topk_hits = torch.FloatTensor(topk_hits).cuda()
    n_docs = torch.FloatTensor([len(answers_list)]).cuda()
    torch.distributed.all_reduce(topk_hits, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(n_docs, torch.distributed.ReduceOp.SUM)

    if torch.distributed.get_rank() == 0:
        topk_hits = topk_hits / n_docs
        print(string_prefix)
        for i in args.report_topk_accuracies:
            print_rank_0("top-{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
        print("\n")

answers_list = []
for i in range(len(dataset)):
    answers_list.append([
        x['has_answer'] for x in dataset[i]['encoder_ids']
    ])

compute_topk_recall(answers_list)
# def topk_accuracy(ctxs, args):
#     if args.sort_by_score:
#         ctxs = sorted(ctxs, key=lambda d: d['score'], reverse=True)

#     relevant20 = [x for x in ctxs[:20] if x['has_answer']]
#     relevant100 = [x for x in ctxs[:100] if x ['has_answer']]
#     return len(relevant20)/20, len(relevant100)/100

# accs20, accs100 = [], []

# for i in range(len(dataset)):
#     a20, a100 = topk_accuracy(dataset[i]['encoder_ids'], args)
#     accs20.append(a20)
#     accs100.append(a100)

# print(f"Acc@20 = {round(np.mean(accs20)*100, 5)}")
# print(f"Acc@100 = {round(np.mean(accs100)*100, 5)}")