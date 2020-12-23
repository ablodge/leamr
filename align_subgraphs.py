import math
import sys

from amr_utils.alignments import write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader

from display import Display
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data


def perplexity(align_model, eval_amrs, eval_alignments):
    perplexity = 0.0
    N = 0
    for amr in eval_amrs:
        tally = 0
        for align in eval_alignments[amr.id]:
            logp = align_model.logp(amr, eval_alignments, align)
            if math.isinf(logp): continue
            tally -= logp / math.log(2.0)
        perplexity += math.pow(2.0, tally / len(eval_alignments[amr.id]))
        N += 1
    perplexity /= N
    print(f'Avg Perplexity: {perplexity}')


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



def main():

    amr_file = sys.argv[1]

    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)
    # amrs = amrs[:10000]
    add_nlp_data(amrs, amr_file)

    eval_amrs = None
    eval_amr_file = None
    eval_alignments = None
    if len(sys.argv)>2:
        eval_amr_file = sys.argv[2]
        eval_amrs = cr.load(eval_amr_file, remove_wiki=True)
        add_nlp_data(eval_amrs, eval_amr_file)

    align_model = Subgraph_Model(amrs, align_duplicates=True)

    iters = 3

    all_alignments = []
    all_eval_alignments = []
    alignments = None

    for i in range(iters):
        print(f'Epoch {i}')
        print('train')
        alignments = align_model.align_all(amrs)
        align_model.update_parameters(amrs, alignments)
        perplexity(align_model, amrs, alignments)

        Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.subgraphs.epoch{i}.html')

        align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.epoch{i}.json'
        print(f'Writing subgraph alignments to: {align_file}')
        write_to_json(align_file, alignments)
        all_alignments.append(alignments)

        if eval_amrs:
            print('eval')
            eval_alignments = align_model.align_all(eval_amrs)
            perplexity(align_model, eval_amrs, eval_alignments)
            all_eval_alignments.append(eval_alignments)

    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.json'
    print(f'Writing subgraph alignments to: {align_file}')
    write_to_json(align_file, alignments)

    print('align subgraphs (train)', coverage(amrs, alignments))

    if eval_amrs:
        print('align subgraphs (eval)', coverage(eval_amrs, eval_alignments))
        align_file = eval_amr_file.replace('.txt', '') + f'.subgraph_alignments.json'
        print(f'Writing subgraph alignments to: {align_file}')
        write_to_json(align_file, eval_alignments)
        Display.style(eval_amrs[:100], eval_amr_file.replace('.txt', '') + f'.subgraphs.html')

    log_rare_alignments(amr_file, align_model, amrs, alignments)
    if eval_amrs:
        log_rare_alignments(eval_amr_file, align_model, eval_amrs, eval_alignments)


if __name__=='__main__':
    main()