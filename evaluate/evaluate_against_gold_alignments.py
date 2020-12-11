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

    print('coverage:',coverage(amrs, subgraph_alignments))
    print('gold coverage:', coverage(amrs, gold_alignments))

    correct = 0
    total = 0
    precision = 0
    precision_total = 0
    recall = 0
    recall_total = 0
    span_correct = 0
    span_total = 0
    for amr in amrs:
        if amr.id not in gold_alignments:
            continue
        gold_spans = ['B' for tok in amr.tokens]
        for align in gold_alignments[amr.id]:
            if len(align.tokens)>1:
                gold_spans[align.tokens[0]] = 'B'
                for t in align.tokens[1:]:
                    gold_spans[t] = 'I'
        pred_spans = ['B' for tok in amr.tokens]
        for align in subgraph_alignments[amr.id]:
            if len(align.tokens) > 1:
                pred_spans[align.tokens[0]] = 'B'
                for t in align.tokens[1:]:
                    pred_spans[t] = 'I'
        for t1,t2 in zip(pred_spans, gold_spans):
            if t1==t2:
                span_correct+=1
            span_total+=1
        for n in amr.nodes:
            gold_align = amr.get_alignment(gold_alignments, node_id=n)
            pred_align = amr.get_alignment(subgraph_alignments, node_id=n)
            if not gold_align:
                continue
            union = set(gold_align.tokens).union(pred_align.tokens)
            intersec = set(gold_align.tokens).intersection(pred_align.tokens)
            if len(union)==0:
                continue

            # correct += len(intersec)/len(union)
            if len(intersec)>0:
                correct+=1
            else:
                print(' '.join(amr.tokens[t] for t in pred_align.tokens), '!=',
                      ' '.join(amr.tokens[t] for t in gold_align.tokens))
                print()
            total+=1
            # if is_aligned1 and is_aligned2:
            #     precision+=1
            #     recall+=1
            # if is_aligned1:
            #     precision_total+=1
            # if is_aligned2:
            #     recall_total+=1
    print('span accuracy:', span_correct, ' / ', span_total, ' = ', f'{100 * span_correct / span_total:.2f}%')
    print('partial accuracy:', correct, ' / ', total, ' = ', f'{100*correct/total:.2f}%')
    # print(precision, ' / ', precision_total, ' = ', precision / precision_total)
    # print(recall, ' / ', recall_total, ' = ', recall / recall_total)




if __name__=='__main__':
    main()