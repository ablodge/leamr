import sys

from amr_utils.amr_readers import AMR_Reader
from models.reentrancy_model import Reentrancy_Model
from models.relation_model import Relation_Model
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-T','--train', required=True, type=str,
                    help='train AMR file (must have nlp data)')
parser.add_argument('-M','--model', required=True, type=str,
                    help='train alignment file')
parser.add_argument('-t','--test', type=str, required=True,
                    help='test AMR file (must have nlp data)')
args = parser.parse_args()


def main():
    train_amr_file = args.train
    train_align_file = args.model
    unaligned_amr_file = args.test

    reader = AMR_Reader()
    train_amrs = reader.load(train_amr_file, remove_wiki=True)
    add_nlp_data(train_amrs, train_amr_file)

    eval_amrs = reader.load(unaligned_amr_file, remove_wiki=True)
    add_nlp_data(eval_amrs, unaligned_amr_file)

    # subgraphs
    align_file = train_align_file.replace('.txt','')+'.subgraph_alignments.json'
    train_alignments = reader.load_alignments_from_json(align_file, train_amrs)

    align_model = Subgraph_Model(train_amrs, align_duplicates=True)
    align_model.update_parameters(train_amrs, train_alignments)

    sub_alignments = align_model.align_all(eval_amrs)
    align_file = unaligned_amr_file.replace('.txt', '') + f'.subgraph_alignments.json'
    print(f'Writing subgraph alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, sub_alignments)
    
    # relations
    align_file = train_align_file.replace('.txt', '') + '.relation_alignments.json'
    train_alignments = reader.load_alignments_from_json(align_file, train_amrs)

    align_model = Relation_Model(train_amrs, sub_alignments)
    align_model.update_parameters(train_amrs, train_alignments)

    rel_alignments = align_model.align_all(eval_amrs)
    align_file = unaligned_amr_file.replace('.txt', '') + f'.relation_alignments.json'
    print(f'Writing relation alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, rel_alignments)

    # reentrancies
    align_file = train_align_file.replace('.txt', '') + '.reentrancy_alignments.json'
    train_alignments = reader.load_alignments_from_json(align_file, train_amrs)

    align_model = Reentrancy_Model(train_amrs, sub_alignments, rel_alignments)
    align_model.update_parameters(train_amrs, train_alignments)

    reent_alignments = align_model.align_all(eval_amrs)
    align_file = unaligned_amr_file.replace('.txt', '') + f'.reentrancy_alignments.json'
    print(f'Writing reentrancy alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, reent_alignments)


if __name__=='__main__':
    main()