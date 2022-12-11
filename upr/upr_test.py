from abc import ABC
import json
import random
import pathlib
import argparse
from torch.utils.data import Dataset
from utils import print_rank_0
from IPython import embed
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
import re

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=pathlib.Path, 
                    required=True, help='Path to dataset.')

args = parser.parse_args()

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

dataset = OpenQADataset("entity_linking",
                        "open-domain retrieval",
                        str(args.dataset),
                        1.)

print(f"Question: {dataset[2332]['question']}")

text = dataset[2332]['question']
context_left, mention = text.split('[START_ENT]')
mention, context_right = mention.split('[END_ENT]')

text = context_left + ' <extra_id_0> ' + context_right

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

embed()