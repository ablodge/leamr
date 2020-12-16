import sys

from amr_utils.alignments import write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader

from display import Display
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data





def coverage(amrs, alignments):
    coverage_count = 0
    total = 0
    for amr in amrs:
        for n in amr.nodes:
            align = amr.get_alignment(alignments, node_id=n)
            if align:
                coverage_count+=1
            total+=1
    return f'{100*coverage_count/total:.2f}%'



def main():

    amr_file = sys.argv[1]

    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)
    # amrs = amrs[:100]
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

    print('align subgraphs (train)', coverage(amrs, alignments))

    differences = {}
    for amr in amrs:
        for span in amr.spans:
            compare = []
            aligns = []
            for alignments in all_alignments:
                align = amr.get_alignment(alignments, token_id=span[0])
                aligns.append(align)
                compare.append(tuple(sorted(align.nodes)))
            if not all(compare[i]==compare[0] for i in range(len(compare))):
                if amr.id not in differences:
                    differences[amr.id] = []
                differences[amr.id].append((amr, ' '.join(amr.tokens[t] for t in span))+ tuple(a for a in aligns))
    with open('data/compare.tsv', 'w+', encoding='utf8') as f:
        for amr_id in differences:
            amr = differences[amr_id][0][0]
            f.write(f'{amr_id}\t{" ".join(amr.tokens)}\n')
            f.write(amr.graph_string())
            for _, token_label, align1, align2, align3 in differences[amr_id]:
                f.write('\t'.join([','.join(str(t) for t in align1.tokens), token_label]))
                for align in [align1, align2, align3]:
                    f.write('\t')
                    if align.nodes:
                        f.write('_'.join([amr.nodes[n] for n in align.nodes]))
                    else:
                        f.write('_')
                f.write('\n')
            f.write('\n')

    eval_amr_file = sys.argv[2]

    eval_amrs = cr.load(eval_amr_file, remove_wiki=True)
    add_nlp_data(eval_amrs, eval_amr_file)

    eval_alignments = align_model.align_all(eval_amrs)
    print('align subgraphs (eval)', coverage(eval_amrs, eval_alignments))



if __name__=='__main__':
    main()