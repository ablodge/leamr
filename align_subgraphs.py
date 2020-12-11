import sys

from amr_utils.alignments import write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader

from display import Display
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data


def main():

    amr_file = sys.argv[1]

    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)
    amrs = amrs[:10000]
    add_nlp_data(amrs, amr_file)

    align_model = Subgraph_Model(amrs, ignore_duplicates=True)
    iters = 3

    # align_file = sys.argv[1].replace('.txt','')+'.subgraph_alignments.tmp2.json'
    # alignments = load_from_json(align_file, amrs)
    # align_model.update_parameters(amrs, alignments)

    all_alignments = []
    alignments = None

    for i in range(iters):
        print(f'Epoch {i}')
        alignments = align_model.align_all(amrs)
        align_model.update_parameters(amrs, alignments)

        Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.subgraphs.epoch{i}.html')

        align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.epoch{i}.tmp1.json'
        print(f'Writing subgraph alignments to: {align_file}')
        write_to_json(align_file, alignments)
        all_alignments.append(alignments)

    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.tmp1.json'
    print(f'Writing subgraph alignments to: {align_file}')
    write_to_json(align_file, alignments)
    all_alignments.append(alignments)


if __name__=='__main__':
    main()