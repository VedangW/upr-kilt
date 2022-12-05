import argparse
from upr.utils.dpr_wiki_dataset import get_open_retrieval_wiki_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--evidence_data_path', type=str, required=True)
args = parser.parse_args()

self.evidence_dataset = get_open_retrieval_wiki_dataset(args=self.args,
                                                        tokens_encode_func=None)