import sys

from amr_utils.amr_readers import AMR_Reader
from models.reentrancy_model import Reentrancy_Model
from models.relation_model import Relation_Model
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subgraph-model', type=str, required=True,
                    help='model parameters file for subgraph aligner')
parser.add_argument('--relation-model', type=str, required=True,
                    help='model parameters file for relation aligner')
parser.add_argument('--reentrancy-model', type=str, required=True,
                    help='model parameters file for reentrancy aligner')
parser.add_argument('-t','--test', type=str, required=True,
                    help='test AMR file (must have nlp data)')
args = parser.parse_args()


def main():
    unaligned_amr_file = args.test

    reader = AMR_Reader()

    eval_amrs = reader.load(unaligned_amr_file, remove_wiki=True)
    add_nlp_data(eval_amrs, unaligned_amr_file)

    # subgraphs
    print(f'Loading model: {args.subgraph_model}')
    subgraph_model = Subgraph_Model.load_model(args.subgraph_model)

    sub_alignments = subgraph_model.align_all(eval_amrs)
    align_file = unaligned_amr_file.replace('.txt', '') + f'.subgraph_alignments.json'
    print(f'Writing subgraph alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, sub_alignments)
    
    # relations
    print(f'Loading model: {args.relation_model}')
    rel_model = Relation_Model.load_model(args.relation_model)
    rel_model.subgraph_alignments = sub_alignments

    rel_alignments = rel_model.align_all(eval_amrs)
    align_file = unaligned_amr_file.replace('.txt', '') + f'.relation_alignments.json'
    print(f'Writing relation alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, rel_alignments)

    # reentrancies
    print(f'Loading model: {args.reentrancy_model}')
    reent_model = Reentrancy_Model.load_model(args.reentrancy_model)
    reent_model.subgraph_alignments = sub_alignments
    reent_model.relation_alignments = rel_alignments

    reent_alignments = reent_model.align_all(eval_amrs)
    align_file = unaligned_amr_file.replace('.txt', '') + f'.reentrancy_alignments.json'
    print(f'Writing reentrancy alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, reent_alignments)


if __name__=='__main__':
    main()