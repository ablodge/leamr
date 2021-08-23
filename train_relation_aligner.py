import sys

from amr_utils.amr_readers import AMR_Reader
from evaluate.utils import perplexity, evaluate_relations
from models.relation_model import Relation_Model
from nlp_data import add_nlp_data


USE_GOLD_SUBGRAPHS = False


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-T','--train', required=True, type=str,
                    help='train AMR file (must have nlp data, subgraph alignments)')
parser.add_argument('-t','--test', type=str, nargs=2,
                    help='2 arguments: test AMR file and gold alignments file (must have nlp data, subgraph alignments)')
parser.add_argument('--iter', type=int, default=5,
                    help='number of iterations to train model')
parser.add_argument('--save-model', type=str,
                    help='params file to store the trained model')
parser.add_argument('--load-model', type=str,
                    help='params file to load model')
args = parser.parse_args()

def report_progress(amr_file, alignments, reader, epoch=None):
    epoch = '' if epoch is None else f'.epoch{epoch}'
    align_file = amr_file.replace('.txt', '') + f'.relation_alignments{epoch}.json'
    print(f'Writing relation alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, alignments)


def main():
    amr_file = args.train

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    add_nlp_data(amrs, amr_file)

    align_file = amr_file.replace('.txt', '') + '.subgraph_alignments.json'
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)
    # amrs = amrs[:1000]

    eval_amr_file, eval_amrs, gold_eval_alignments = None, None, None
    if args.test:
        eval_amr_file, eval_align_file = args.test
        eval_amrs = reader.load(eval_amr_file, remove_wiki=True)
        add_nlp_data(eval_amrs, eval_amr_file)
        gold_eval_alignments = reader.load_alignments_from_json(eval_align_file, eval_amrs)
        eval_amr_ids = {amr.id for amr in eval_amrs}
        amrs = [amr for amr in amrs if amr.id not in eval_amr_ids]

        align_file = eval_amr_file.replace('.txt', '') + '.subgraph_alignments.gold.json'
        gold_subgraph_alignments = reader.load_alignments_from_json(align_file, eval_amrs)
        align_file = eval_amr_file.replace('.txt', '') + '.subgraph_alignments.json'
        pred_subgraph_alignments = reader.load_alignments_from_json(align_file, eval_amrs)
        if USE_GOLD_SUBGRAPHS:
            pred_subgraph_alignments = gold_subgraph_alignments
        for amr_id in pred_subgraph_alignments:
            subgraph_alignments[amr_id] = pred_subgraph_alignments[amr_id]
        for amr in eval_amrs:
            spans = [align.tokens for align in pred_subgraph_alignments[amr.id] if align.type=='subgraph']
            amr.spans = spans


    if args.load_model:
        print('Loading model from:', args.load_model)
        align_model = Relation_Model.load_model(args.load_model)
    else:
        align_model = Relation_Model(amrs, subgraph_alignments)

    iters = args.iter

    alignments = None
    for i in range(iters):
        print(f'Epoch {i}: Training data')
        alignments = align_model.align_all(amrs)
        align_model.update_parameters(amrs, alignments)
        report_progress(amr_file, alignments, reader, epoch=i)
        perplexity(align_model, amrs, alignments)
        print()

        if eval_amrs:
            print(f'Epoch {i}: Evaluation data')
            eval_alignments = align_model.align_all(eval_amrs)
            perplexity(align_model, eval_amrs, eval_alignments)
            evaluate_relations(eval_amrs, eval_alignments, gold_eval_alignments, pred_subgraph_alignments, gold_subgraph_alignments)
            report_progress(eval_amr_file, eval_alignments, reader, epoch=i)
            print()


    report_progress(amr_file, alignments, reader)

    if args.save_model:
        align_model.save_model(args.save_model)
        print('Saving model to:', args.save_model)



if __name__ == '__main__':
    main()