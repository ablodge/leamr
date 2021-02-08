import sys

from amr_utils.amr_readers import AMR_Reader

from evaluate.utils import evaluate, evaluate_relations, evaluate_reentrancies, evaluate_duplicates
from nlp_data import add_nlp_data


def main():
    amr_file = sys.argv[1]
    align_file = sys.argv[2]
    gold_file = sys.argv[3]

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    add_nlp_data(amrs, amr_file)

    alignments = reader.load_alignments_from_json(align_file, amrs)
    gold_alignments = reader.load_alignments_from_json(gold_file, amrs)
    pred_subgraph_alignments = reader.load_alignments_from_json(align_file.replace('relation_','subgraph_'), amrs)
    gold_subgraph_alignments = reader.load_alignments_from_json(gold_file.replace('relation_','subgraph_'), amrs)

    # Display.style([amr for amr in amrs if amr.id in gold_alignments],
    #               gold_file.replace('.json', '') + f'.html',
    #               gold_alignments)

    if len(amrs)!=len(alignments):
        amrs = [amr for amr in amrs if amr.id in alignments and amr.id in gold_alignments]
    evaluate(amrs, alignments, gold_alignments, mode='edges')
    # evaluate_relations(amrs, alignments, gold_alignments, pred_subgraph_alignments, gold_subgraph_alignments)
    # evaluate_reentrancies(amrs, alignments, gold_alignments)
    # evaluate_duplicates(amrs, alignments, gold_alignments)


if __name__=='__main__':
    main()