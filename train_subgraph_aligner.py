import sys

from amr_utils.alignments import load_from_json
from amr_utils.amr_readers import AMR_Reader

from evaluate.utils import evaluate, perplexity, evaluate_duplicates
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-T','--train', required=True, type=str,
                    help='train AMR file (must have nlp data)')
parser.add_argument('-t','--test', type=str, nargs=2,
                    help='2 arguments: test AMR file and gold alignments file (must have nlp data)')
args = parser.parse_args()

def report_progress(amr_file, alignments, reader, epoch=None):
    epoch = '' if epoch is None else f'.epoch{epoch}'
    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments{epoch}.json'
    print(f'Writing subgraph alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, alignments)

def main():

    amr_file = args.train

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    add_nlp_data(amrs, amr_file)

    eval_amr_file, eval_amrs, gold_eval_alignments = None, None, None
    if args.test:
        eval_amr_file, eval_align_file = args.test
        eval_amrs = reader.load(eval_amr_file, remove_wiki=True)
        add_nlp_data(eval_amrs, eval_amr_file)
        gold_eval_alignments = load_from_json(eval_align_file, eval_amrs, unanonymize=True)
        eval_amr_ids = {amr.id for amr in eval_amrs}
        amrs = [amr for amr in amrs if amr.id not in eval_amr_ids]
    # amrs = amrs[:1000]

    align_model = Subgraph_Model(amrs, align_duplicates=True)

    iters = 5

    for i in range(iters):
        print(f'Epoch {i}: Training data')
        alignments = align_model.align_all(amrs)
        align_model.update_parameters(amrs, alignments)
        perplexity(align_model, amrs, alignments)
        report_progress(amr_file, alignments, reader, epoch=i)
        print()

        if eval_amrs:
            print(f'Epoch {i}: Evaluation data')
            eval_alignments = align_model.align_all(eval_amrs)
            perplexity(align_model, eval_amrs, eval_alignments)
            evaluate(eval_amrs, eval_alignments, gold_eval_alignments)
            evaluate_duplicates(eval_amrs, eval_alignments, gold_eval_alignments)
            report_progress(eval_amr_file, eval_alignments, reader, epoch=i)
            print()


if __name__=='__main__':
    main()