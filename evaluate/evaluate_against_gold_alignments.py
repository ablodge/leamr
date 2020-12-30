import sys

from amr_utils.alignments import load_from_json
from amr_utils.amr_readers import JAMR_AMR_Reader

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
    align_file = sys.argv[2]
    gold_file = sys.argv[3]

    amr_reader = JAMR_AMR_Reader()
    amrs = amr_reader.load(amr_file, remove_wiki=True)
    add_nlp_data(amrs, amr_file)

    subgraph_alignments = load_from_json(align_file, amrs)
    gold_alignments = load_from_json(gold_file, amrs)

    print('coverage:', coverage(amrs, subgraph_alignments))
    print('gold coverage:', coverage(amrs, gold_alignments))

    correct = 0
    total = 0
    span_correct = 0
    span_total = 0
    for amr in amrs:
        if amr.id not in gold_alignments:
            continue
        for tok in range(len(amr.tokens)):
            gold_align = amr.get_alignment(gold_alignments, token_id=tok)
            pred_align = amr.get_alignment(subgraph_alignments, token_id=tok)
            union = set(gold_align.nodes).union(pred_align.nodes)
            intersec = set(gold_align.nodes).intersection(pred_align.nodes)
            # if set(pred_align.nodes)!=set(gold_align.nodes):
            #     token1 = ' '.join(amr.tokens[t] for t in gold_align.tokens)
            #     token2 = ' '.join(amr.tokens[t] for t in span)
            #     nodes1 = '_'.join(amr.nodes[n] for n in pred_align.nodes)
            #     if not nodes1:
            #         nodes1 = '_'
            #     nodes2 = '_'.join(amr.nodes[n] for n in gold_align.nodes)
            #     if not nodes2:
            #         nodes2 = '_'
            #     print(amr.id, span, token1, token2, nodes1, '!=', nodes2)
            #     print()
            total+=1
            if len(gold_align.nodes)==0 and len(pred_align.nodes)==0:
                    correct+=1
            else:
                correct += len(intersec)/len(union)

        for span in amr.spans:
            gold_align = amr.get_alignment(gold_alignments, token_id=span[0])
            if gold_align.tokens == span:
                span_correct+=1
            span_total+=1

    print('partial accuracy:', correct, ' / ', total, ' = ', f'{100*correct/total:.2f}%')
    print('span accuracy:', span_correct, ' / ', span_total, ' = ', f'{100 * span_correct / span_total:.2f}%')




if __name__=='__main__':
    main()