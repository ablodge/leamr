import math
import sys

from amr_utils.alignments import AMR_Alignment


def table_to_latex(table, row_labels, column_labels):
    sep = ' & '
    end = ' \\\\'
    latex = sep.join(column_labels)+end+'\n'
    for row_label, row in zip(row_labels, table):
        latex += row_label+sep+sep.join(f'{t:.2f}' if isinstance(t, float) else str(t) for t in row)+end+'\n'
    return latex

def table_to_excel(table, row_labels, column_labels):
    sep = '\t'
    end = ''
    s = sep.join(column_labels)+end+'\n'
    for row_label, row in zip(row_labels, table):
        s += row_label+sep+sep.join(f'{t:.2f}' if isinstance(t, float) else str(t) for t in row)+end+'\n'
    return s


def coverage(amrs, alignments, mode='nodes'):
    coverage_count = 0
    total = 0
    for amr in amrs:
        for n in getattr(amr, mode):
            align = amr.get_alignment(alignments, node_id=n) \
                if mode=='nodes' else amr.get_alignment(alignments, edge=n)
            if align:
                coverage_count+=1
            total+=1
    return f'{100*coverage_count/total:.2f}%'


SEP = '\t'

def evaluate(amrs, pred_alignments, gold_alignments, mode='nodes'):
    print('pred coverage:', coverage(amrs, pred_alignments, mode))
    print('gold coverage:', coverage([amr for amr in amrs if amr.id in gold_alignments], gold_alignments, mode))

    span_tp = 0
    span_recall_total = 0
    span_prec_total = 0

    subgraph_tp = 0
    subgraph_recall_total = 0
    subgraph_prec_total = 0

    partial_subgraph_tp = 0

    for amr in amrs:
        if amr.id not in gold_alignments:
            continue

    for amr in amrs:
        if amr.id not in gold_alignments:
            continue

        for gold_align in gold_alignments[amr.id]:
            if gold_align.type.startswith('dupl'): continue
            pred_align = amr.get_alignment(pred_alignments, token_id=gold_align.tokens[0])

            if gold_align or pred_align:
                span_recall_total += 1
                if gold_align.tokens==pred_align.tokens:
                    span_tp += 1

            # if set(getattr(pred_align, mode))!=set(getattr(gold_align, mode)):
            #     token1 = ' '.join(amr.tokens[t] for t in pred_align.tokens)
            #     token2 = ' '.join(amr.tokens[t] for t in gold_align.tokens)
            #     nodes1 = '_'.join(amr.nodes[n] for n in pred_align.nodes) if pred_align.nodes else '_'
            #     nodes2 = '_'.join(amr.nodes[n] for n in gold_align.nodes) if gold_align.nodes else '_'
            #     edges1 = '_'.join(r for s,r,t in pred_align.edges) if pred_align.edges else '_'
            #     edges2 = '_'.join(r for s,r,t in gold_align.edges) if gold_align.edges else '_'
            #     gold_align_label = f'{gold_align.tokens}->{gold_align.edges}'
            #     pred_align_label = f'{pred_align.tokens}->{pred_align.edges}'
            #     print(amr.id, token1, token2, edges1, '!=', edges2, pred_align_label, gold_align_label, )
            #     print()

            if not gold_align: continue

            subgraph_recall_total += 1
            if gold_align.tokens==pred_align.tokens:
                if set(getattr(gold_align, mode)) == set(getattr(pred_align, mode)):# and gold_align.type==pred_align.type:
                    subgraph_tp += 1

            pred_aligns = [align for align in pred_alignments[amr.id] if any(t in align.tokens for t in gold_align.tokens) and align]
            max_score = 0
            for align in pred_aligns:
                n_inter = set(getattr(gold_align, mode)).intersection(getattr(align, mode))
                n_union = set(getattr(gold_align, mode)).union(getattr(align, mode))
                t_inter = set(gold_align.tokens).intersection(align.tokens)
                t_union = set(gold_align.tokens).union(align.tokens)
                score = (len(n_inter)/len(n_union))*(len(t_inter)/len(t_union))
                if score>max_score:
                    max_score = score
            partial_subgraph_tp += max_score


        for pred_align in pred_alignments[amr.id]:
            if pred_align.type.startswith('dupl'): continue
            gold_align = amr.get_alignment(gold_alignments, token_id=pred_align.tokens[0])
            if gold_align or pred_align:
                span_prec_total += 1

            if not pred_align: continue
            subgraph_prec_total+=1

        for align in gold_alignments[amr.id]+pred_alignments[amr.id]:
            for n in getattr(align, mode):
                if n not in getattr(amr, mode):
                    print('Found Faulty Alignment:', amr.id, n,file=sys.stderr)

    # F1 = 2*(prec*recall)/(prec+recall)
    span_prec = span_tp/span_prec_total if span_prec_total>0 else 0
    span_recall = span_tp/span_recall_total if span_recall_total>0 else 0
    span_f1 = 2*(span_prec*span_recall)/(span_prec+span_recall) if (span_prec+span_recall)>0 else 0
    print(f'Span F1:{SEP}{100 * span_f1:.2f}{SEP}(#gold {span_recall_total})')

    print(f'Score{SEP}Precision{SEP}Recall{SEP}F1')

    subgraph_prec = partial_subgraph_tp / subgraph_prec_total if subgraph_prec_total > 0 else 0
    subgraph_recall = partial_subgraph_tp / subgraph_recall_total if subgraph_recall_total > 0 else 0
    subgraph_f1 = 2 * (subgraph_prec * subgraph_recall) / (subgraph_prec + subgraph_recall) if (subgraph_prec + subgraph_recall) > 0 else 0
    print(f'Partial Align:{SEP}{100 * subgraph_prec:.2f}{SEP}{100 * subgraph_recall:.2f}{SEP}{100 * subgraph_f1:.2f}{SEP}(#gold {subgraph_recall_total})')

    subgraph_prec = subgraph_tp / subgraph_prec_total if subgraph_prec_total>0 else 0
    subgraph_recall = subgraph_tp / subgraph_recall_total if subgraph_recall_total>0 else 0
    subgraph_f1 = 2 * (subgraph_prec * subgraph_recall) / (subgraph_prec + subgraph_recall) if (subgraph_prec + subgraph_recall)>0 else 0
    print(f'Exact Align:{SEP}{100 * subgraph_prec:.2f}{SEP}{100 * subgraph_recall:.2f}{SEP}{100 * subgraph_f1:.2f}{SEP}(#gold {subgraph_recall_total})')


def evaluate_relations(amrs, pred_rel_alignments, gold_rel_alignments, pred_sub_alignments, gold_sub_alignments):
    pred_single_alignments = {}
    pred_argstruct_alignments = {}
    gold_single_alignments = {}
    gold_argstruct_alignments = {}
    for amr in amrs:
        pred_single_alignments[amr.id] = []
        pred_argstruct_alignments[amr.id] = []
        for align in pred_rel_alignments[amr.id]:
            sub_align = amr.get_alignment(gold_sub_alignments, token_id=align.tokens[0])
            if sub_align.nodes:
                align.edges = [e for e in align.edges if not (e[0] in sub_align.nodes and e[-1] in sub_align.nodes)]
                pred_argstruct_alignments[amr.id].append(align)
            else:
                pred_single_alignments[amr.id].append(align)
        gold_single_alignments[amr.id] = []
        gold_argstruct_alignments[amr.id] = []
        for align in gold_rel_alignments[amr.id]:
            sub_align = amr.get_alignment(gold_sub_alignments, token_id=align.tokens[0])
            if sub_align.nodes:
                align.edges = [e for e in align.edges if not (e[0] in sub_align.nodes and e[-1] in sub_align.nodes)]
                gold_argstruct_alignments[amr.id].append(align)
            else:
                gold_single_alignments[amr.id].append(align)

    print('single relations')
    evaluate(amrs, pred_single_alignments, gold_single_alignments, mode='edges')
    print('argument structures')
    evaluate(amrs, pred_argstruct_alignments, gold_argstruct_alignments, mode='edges')
    print('total')
    evaluate(amrs, pred_rel_alignments, gold_rel_alignments, mode='edges')


def evaluate_reentrancies(amrs, pred_alignments, gold_alignments):
    types = set()
    for amr in amrs:
        for align in pred_alignments[amr.id]:
            type = align.type.split(':')[-1]
            types.add(type)
        for align in gold_alignments[amr.id]:
            type = align.type.split(':')[-1]
            types.add(type)
    pred_alignments2 = {type: {} for type in types}
    gold_alignments2 = {type: {} for type in types}
    for amr in amrs:
        for type in types:
            pred_aligns = [align for align in pred_alignments[amr.id] if align.type==f'reentrancy:{type}']
            gold_aligns = [align for align in gold_alignments[amr.id] if align.type == f'reentrancy:{type}']
            if pred_aligns or gold_aligns:
                pred_alignments2[type][amr.id] = pred_aligns
                gold_alignments2[type][amr.id] = gold_aligns

    for type in sorted(types):
        print(type)
        evaluate(amrs, pred_alignments2[type], gold_alignments2[type], mode='edges')
    print('total')
    evaluate(amrs, pred_alignments, gold_alignments, mode='edges')


def evaluate_duplicates(amrs, pred_alignments, gold_alignments):
    print('duplicates')
    duplicate_alignments = {}
    gold_duplicate_alignments = {}

    for amr in amrs:
        duplicate_alignments[amr.id] = []
        dupicates = {}
        for align in pred_alignments[amr.id]:
            if align.type.startswith('dupl'):
                span = tuple(align.tokens)
                nodes = set(align.nodes)
                if span not in dupicates:
                    dupicates[span] = set()
                dupicates[span].update(nodes)
        duplicate_alignments[amr.id] = [AMR_Alignment(type='subgraph:dupl', tokens=list(span), nodes=list(dupicates[span])) for span in dupicates]

        gold_duplicate_alignments[amr.id] = []
        dupicates = {}
        for align in gold_alignments[amr.id]:
            if align.type.startswith('dupl'):
                span = tuple(align.tokens)
                nodes = set(align.nodes)
                if span not in dupicates:
                    dupicates[span] = set()
                dupicates[span].update(nodes)
        gold_duplicate_alignments[amr.id] = [AMR_Alignment(type='subgraph:dupl', tokens=list(span), nodes=list(dupicates[span])) for span in dupicates]

    evaluate(amrs, duplicate_alignments, gold_duplicate_alignments, mode='nodes')


def perplexity(align_model, eval_amrs, eval_alignments):
    perplexity = 0.0
    N = 0
    for amr in eval_amrs:
        if not eval_alignments[amr.id]:
            continue
        tally = 0
        for align in eval_alignments[amr.id]:
            logp = align_model.logp(amr, eval_alignments, align)
            if math.isinf(logp): continue
            tally -= logp / math.log(2.0)
        perplexity += math.pow(2.0, tally / len(eval_alignments[amr.id]))
        N += 1
    perplexity /= N
    print(f'Avg Perplexity: {perplexity}')


def log_rare_alignments(file, align_model, amrs, alignments):
    amr_ids = {amr.id:amr for amr in amrs}

    rare_alignments = {}
    for amr in amrs:
        for align in alignments[amr.id]:
            if align.type == 'dupl-subgraph': continue
            logp = align_model.trans_logp(amr, align)
            p = math.exp(logp)
            if p < 0.001:
                if amr.id not in rare_alignments:
                    rare_alignments[amr.id] = []
                rare_alignments[amr.id].append((align))
    with open(file.replace('.txt', '.log_rare.txt'), 'w+', encoding='utf8') as f:
        for amr_id in rare_alignments:
            amr = amr_ids[amr_id]
            f.write(f'{amr_id}\t{" ".join(amr.tokens)}\n')
            f.write(amr.graph_string())
            for align in rare_alignments[amr_id]:
                token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
                subgraph_label = '_'.join(amr.nodes[n] for n in align.nodes) if align.nodes else '_'
                f.write(f'{token_label} -> {subgraph_label}\n')
            f.write('\n')

def log_alignment_changes(file, amrs, all_alignments):
    differences = {}
    for amr in amrs:
        for span in amr.spans:
            compare = []
            aligns = []
            for alignments in all_alignments:
                align = amr.get_alignment(alignments, token_id=span[0])
                aligns.append(align)
                compare.append(tuple(sorted(align.nodes)))
            if not all(compare[i] == compare[0] for i in range(len(compare))):
                if amr.id not in differences:
                    differences[amr.id] = []
                differences[amr.id].append((amr, ' '.join(amr.tokens[t] for t in span), aligns))
    with open(file.replace('.txt','.log_alignment_changes.txt'), 'w+', encoding='utf8') as f:
        for amr_id in differences:
            amr = differences[amr_id][0][0]
            f.write(f'{amr_id}\t{" ".join(amr.tokens)}\n')
            f.write(amr.graph_string())
            for _, token_label, aligns in differences[amr_id]:
                f.write('\t'.join([','.join(str(t) for t in aligns[0].tokens), token_label]))
                for align in aligns:
                    f.write('\t')
                    if align.nodes:
                        f.write('_'.join([amr.nodes[n] for n in align.nodes]))
                    else:
                        f.write('_')
                f.write('\n')
            f.write('\n')
