import random
import numpy
import json
import time
import argparse
import os
import shutil
import pickle

import torch
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration

from IPython import embed
from utils import print_rank_0
from utils.openqa_dataset import get_openqa_dataset, get_one_epoch_dataloader
from utils.initialize import initialize_distributed
from utils.dpr_wiki_dataset import get_open_retrieval_wiki_dataset


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


class UnsupervisedPassageReranker():
    def __init__(self, args):
        self.model = None
        self.dataloader = None
        self.dataset = None
        self.evidence_dataset = None

        self.args = args
        self.log_interval = args.log_interval
        # Hard coding the per gpu batch size to 1
        self.batch_size = 1

        self.task_name = args.task_name

        # Empirical distribution arguments
        self.pseudo_count = args.pseudo_count
        self.power = args.power
        self.temperature = args.temperature

        self.load_attributes()
        self.is_main_builder = dist.get_rank() == 0
        self.num_total_builders = dist.get_world_size()

        self.temp_dir_name = os.path.join(args.reranker_output_dir, '_tmp_reranker')

    def load_attributes(self):
        print_rank_0("Loading {} weights".format(self.args.hf_model_name))
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.hf_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.hf_model_name,
                                                                torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float32)

        for param in self.model.parameters():
            param.requires_grad = False

        if self.args.use_gpu:
            self.model = self.model.cuda()

        self.use_priors = self.args.use_priors
        self.total_passages = self.args.total_passages
        self.counts_path = self.args.counts_path

        print_rank_0("Loaded {} weights".format(self.args.hf_model_name))

        # disable dropout
        self.model.eval()

        self.evidence_dataset = get_open_retrieval_wiki_dataset(args=self.args,
                                                                tokens_encode_func=None)

        self.dataset = get_openqa_dataset(self.args.task_name,
                                          self.args.retriever_topk_passages_path,
                                          sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        self.iteration = self.total_processed = 0

    def track_and_report_progress(self, batch_size):
        """Utility function for tracking progress"""
        self.iteration += 1
        self.total_processed += batch_size * self.num_total_builders
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def do_inference(self):
        reranked_answers_list = []
        original_answers_list = []
        reranked_data = []

        start_time = time.time()

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            masked_question = batch['masked_question'][0]
            all_contexts = batch['encoder_ids'][0][:self.args.topk_passages]

            all_ids = []
            has_answer_list = []

            for i, context in enumerate(all_contexts):
                text, title = self.evidence_dataset.id2text[int(context.get("id"))]
                if self.task_name == 'entity_linking':
                    ids = "{} {} {}. {} {}. {}".format(self.args.verbalizer_head, 
                                                       title, 
                                                       text, 
                                                       self.args.query_verbalizer, 
                                                       masked_question, 
                                                       self.args.verbalizer)
                else:
                    ids = "{} {} {}. {}".format(self.args.verbalizer_head, 
                                            title, 
                                            text, 
                                            self.args.verbalizer)
                all_ids.append(ids)
                has_answer_list.append(context.get('has_answer'))

            input_encoding = self.tokenizer(all_ids,
                                            padding='longest',
                                            max_length=2048,
                                            pad_to_multiple_of=8,
                                            truncation=True,
                                            return_tensors='pt')

            context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

            if self.args.use_gpu:
                context_tensor = context_tensor.cuda()
                attention_mask = attention_mask.cuda()

            decoder_prefix = batch['decoder_ids']
            target_encoding = self.tokenizer(decoder_prefix,
                                             max_length=128,
                                             truncation=True,
                                             return_tensors='pt')

            decoder_prefix_tensor = target_encoding.input_ids
            if self.args.use_gpu:
                decoder_prefix_tensor = decoder_prefix_tensor.cuda()

            decoder_prefix_tensor = torch.repeat_interleave(decoder_prefix_tensor,
                                                            len(context_tensor),
                                                            dim=0)
            sharded_nll_list = []

            for i in range(0, len(context_tensor), self.args.shard_size):
                shard_contexts = all_contexts[i: i + self.args.shard_size]
                encoder_tensor_view = context_tensor[i: i + self.args.shard_size]
                attention_mask_view = attention_mask[i: i + self.args.shard_size]
                decoder_tensor_view = decoder_prefix_tensor[i: i + self.args.shard_size]

                with torch.no_grad():
                    logits = self.model(input_ids=encoder_tensor_view,
                                        attention_mask=attention_mask_view,
                                        labels=decoder_tensor_view).logits

                log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
                nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

                avg_nll = torch.sum(nll, dim=1)
                sharded_nll_list.append(avg_nll)

            topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(context_tensor))
            indexes = indexes.cpu()
            ranked_answers = torch.BoolTensor(has_answer_list)[indexes]

            # Save the essential information to be used for saving the re-ranked information component.
            original_answers_list.append(has_answer_list)
            reranked_answers_list.append(ranked_answers.tolist())

            reordered_context = [all_contexts[i] for i in indexes]

            for i, ctx in enumerate(reordered_context):
                ctx['score'] = topk_scores[i].item()

            item = {"question": batch['question'][0],
                    "answers": batch['answers'][0],
                    "ctxs": reordered_context[:self.args.report_topk_accuracies[-1]]}
            reranked_data.append(item)

            self.track_and_report_progress(batch_size=len(batch['id']))

        end_time = time.time()
        time_taken = (end_time - start_time) / len(reranked_data)
        torch.distributed.barrier()

        print_rank_0("Time taken: {} seconds".format(time_taken))

        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")

        if self.args.merge_shards_and_save:
            self.save_shard(reranked_data)

        del self.model
        # This process signals to finalize its shard and then synchronize with the other processes
        torch.distributed.barrier()

        if self.args.merge_shards_and_save:
            # rank 0 process builds the final copy
            if self.is_main_builder:
                self.merge_shards_and_save()
            # complete building the final copy
            torch.distributed.barrier()

    @staticmethod
    def calculate_topk_hits(scores, max_k):
        top_k_hits = [0] * max_k
        for question_hits in scores:
            best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        return top_k_hits

    def compute_topk_recall(self, answers_list, string_prefix):
        topk_hits = self.calculate_topk_hits(answers_list, max_k=self.args.report_topk_accuracies[-1])

        topk_hits = torch.FloatTensor(topk_hits).cuda()
        n_docs = torch.FloatTensor([len(answers_list)]).cuda()
        torch.distributed.all_reduce(topk_hits, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(n_docs, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            topk_hits = topk_hits / n_docs
            print(string_prefix)
            for i in self.args.report_topk_accuracies:
                print_rank_0("top-{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
            print("\n")

    def save_shard(self, data):
        """
        Save the block data that was created this in this process
        """
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        outpath = os.path.join(self.temp_dir_name, "rank{}.json".format(dist.get_rank()))
        with open(outpath, "w") as writer:
            writer.write(json.dumps(data, indent=4) + "\n")

    def merge_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        shard_names = os.listdir(self.temp_dir_name)
        all_data = []

        for fname in os.listdir(self.temp_dir_name):
            shard_size = 0
            old_size = len(all_data)
            fpath = '{}/{}'.format(self.temp_dir_name, fname)
            with open(fpath, 'r') as f:
                data = json.load(f)
                shard_size = len(data)
                all_data.extend(data)

            assert len(all_data) == old_size + shard_size
            os.remove(fpath)

        # save the consolidated shards
        outpath = os.path.join(self.args.reranker_output_dir, "{}.json".format(self.args.special_suffix))

        with open(outpath, 'w') as writer:
            writer.write(json.dumps(all_data, indent=4) + "\n")

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(all_data)), flush=True)

        # make sure that every single piece of data was embedded
        assert len(all_data) == len(self.dataset)

        shutil.rmtree(self.temp_dir_name, ignore_errors=True)


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='argument-parser')

    group.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher.')

    group.add_argument('--main-port', type=int, default=29500,
                       help='Main port number.')

    group.add_argument('--special-suffix', type=str, default="",
                       help='special suffix extension for saving merged file')

    group.add_argument('--retriever-topk-passages-path', type=str, default="downloads/data/retriever-outputs/nq-dev.json",
                       help='Path of the Top-K passage output file from retriever (.json file)')

    group.add_argument('--topk-passages', type=int, default=1000,
                       help='number of topk passages to select')

    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')

    group.add_argument('--shard-size', type=int, default=16)

    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")

    group.add_argument('--reranker-output-dir', type=str, default="downloads/data/retriever-outputs/",
                       help='Path to save UPR results')

    group.add_argument('--task-name', type=str, default="reranking",
                       help='Name of the task.')

    group.add_argument('--hf-model-name', type=str, default="t5-large",
                       help='Name of the HF model.')

    group.add_argument('--interactive-node', action='store_true',
                       help='If the node is interactive or not')

    group.add_argument('--use-gpu', action='store_true',
                       help='Use GPU or not')

    group.add_argument('--use-bf16', action='store_true',
                       help='Whether to use BF16 data format for the T0/T5 models')

    group.add_argument('--merge-shards-and-save', action='store_true',
                       help='whether to merge individual data shards or not for reranking')

    group.add_argument('--sample-rate', type=float, default=1.,
                       help="Sample rate for the number of examples.")

    group.add_argument('--random-seed', type=int, default=1234,
                       help="Random seed.")

    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia evidence passages file')

    group.add_argument('--verbalizer', type=str, default="Please write a question based on this passage.",
                       help='Prompt string for generating the target tokens')

    group.add_argument('--query-verbalizer', type=str, default="Question: ",
                       help='Prompt string before adding query.')

    group.add_argument('--verbalizer-head', type=str, default="Passage: ",
                       help='The string token used to represent encoder input')

    group.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    group.add_argument('--use-priors', action='store_true', 
                       help='Use priors with generative probs.')

    group.add_argument('--total-passages', type=int, default=21015324, 
                       help='No. of total passages retrieved in trained set')

    group.add_argument('--counts-path', type=str, 
                       help='Path to pickle file storing frequencies of retr. docs. from train set.')

    group.add_argument('--pseudo-count', type=int, default=1)

    group.add_argument('--power', type=float, default=1)

    group.add_argument('--temperature', type=int, default=1)

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args


def main():
    args = get_args()
    set_random_seed(args.random_seed)
    initialize_distributed(args)

    reranker = UnsupervisedPassageReranker(args)
    reranker.do_inference()


if __name__ == "__main__":
    main()
