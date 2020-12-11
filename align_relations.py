import sys

from amr_utils.alignments import write_to_json, load_from_json
from amr_utils.amr_readers import JAMR_AMR_Reader

from display import Display
from models.relation_model import Relation_Model
from nlp_data import add_nlp_data


def main():

    amr_file = sys.argv[1]
    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)

    # amrs = amrs[:1000]

    add_nlp_data(amrs, amr_file)

    align_file = sys.argv[1].replace('.txt', '') + '.subgraph_alignments.all.json'
    subgraph_alignments = load_from_json(align_file, amrs)

    align_model = Relation_Model(amrs, subgraph_alignments)
    iters = 3

    all_alignments = []

    for i in range(iters):
        print(f'Epoch {i}')
        alignments = align_model.align_all(amrs)
        align_model.update_parameters(amrs, alignments)

        Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.relations.no-pretrain{i}.html')

        align_file = amr_file.replace('.txt', '') + f'.relation_alignments{i}.json'
        print(f'Writing subgraph alignments to: {align_file}')
        write_to_json(align_file, alignments)
        all_alignments.append(alignments)

    # all_alignments_diff = {}
    # for amr in amrs:
    #     all_alignments_diff[amr.id] = []
    #     for span in amr.spans:
    #         diff = [amr.get_alignment(alignments, token_id=span[0]).nodes for alignments in all_alignments]
    #         diff = {i:nodes for i,nodes in enumerate(diff)}
    #         if not all(nodes==diff[0] for i,nodes in diff.items()):
    #             readable_diff = [align_model.get_alignment_label(amr, diff[i]) for i in diff]
    #             all_alignments_diff[amr.id].append((span, diff,
    #                                                 ' '.join(amr.lemmas[t] for t in span),
    #                                                 readable_diff))
    # print()

if __name__=='__main__':
    main()