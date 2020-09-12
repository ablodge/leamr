import sys
from collections import Counter

from amr_utils.alignments import load_from_json, convert_alignment_to_subgraph, AMR_Alignment, write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader
from amr_utils.graph_utils import get_rooted_components

from tqdm import tqdm

from display import Display
from models.subgraph_model import Subgraph_Model
from models.utils import align_all
from nlp_data import add_nlp_data
from rule_based.subgraph_rules import is_subgraph, subgraph_fuzzy_align, subgraph_exact_align_english, \
    postprocess_subgraph, postprocess_subgraph_english, clean_subgraph

long_spans = Counter()

TRAIN_CAP = None
# TRAIN_CAP = 100

VERBOSE = False
IGNORE_DUPLICATES = True
ENGLISH = True


def separate_components(amr, align):
    node_labels = [amr.nodes[n] for n in align.nodes]
    if len(node_labels) > 1 and all(node == node_labels[0] for node in node_labels):
        return [AMR_Alignment(type='subgraph', tokens=align.tokens, nodes=[n], amr=amr) for n in align.nodes]
    align_amr = convert_alignment_to_subgraph(align, amr)
    if align_amr is None:
        return [align]
    if is_subgraph(amr, align.nodes):
        return [align]
    components = get_rooted_components(align_amr)
    components = [list(sub.nodes.keys()) for sub in components]
    components = [AMR_Alignment(type='subgraph', tokens=align.tokens, nodes=nodes, amr=amr) for nodes in components]
    return components

def merge_spans(amr, alignments, spans):
    new_alignments = []
    for span in spans:
        new_align = AMR_Alignment(type='subgraph', tokens=span, amr=amr)
        all_nodes = set()
        for i, align in enumerate(alignments[:]):
            if align.tokens[0] in span:
                all_nodes.update(align.nodes)
        new_align.nodes = list(all_nodes)
        new_alignments.append(new_align)
    return new_alignments

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

def resolve_inconsistent_alignments(amrs, subgraph_alignments, candidate_alignments, model):
    for amr in tqdm(amrs, file=sys.stdout):
        for aligns in candidate_alignments[amr.id]:
            if not aligns: continue
            scores = {}
            for i,align in enumerate(aligns):
                scores[i] = model.logp(amr, subgraph_alignments, align)
            best = max(scores, key=lambda x:scores[x])
            new_align = aligns[best]
            add_alignment(amr, subgraph_alignments, new_align)


def add_alignment(amr, alignments, align):
    for j,old_align in enumerate(alignments[amr.id]):
        if old_align.tokens == align.tokens:
            alignments[amr.id][j] = align
            postprocess_subgraph(amr, alignments, align)
            if ENGLISH: postprocess_subgraph_english(amr, alignments, align)
            clean_subgraph(amr, alignments, align)
            return
    raise Exception('Alignment not found!')

def remove_alignment(amr, alignments, align):
    for j,old_align in enumerate(alignments[amr.id]):
        if old_align.tokens == align.tokens:
            alignments[amr.id][j] = AMR_Alignment(type='subgraph', tokens=align.tokens)
            return
    raise Exception('Alignment not found!')


def main():
    amr_file = sys.argv[1]
    align_file = sys.argv[1].replace('.txt','')+'.node_alignments.json'

    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)
    # amrs = [amr for amr in amrs if amr.id == 'bolt12_10465_5592.3']
    if TRAIN_CAP is not None:
        amrs = amrs[:TRAIN_CAP]

    add_nlp_data(amrs, amr_file)

    subgraph_model = Subgraph_Model(amrs, ignore_duplicates=IGNORE_DUPLICATES)

    node_alignments = load_from_json(align_file)
    subgraph_alignments = subgraph_model.get_initial_alignments(amrs)
    multiple_subgraph_alignments = {}
    multiple_span_alignments = {}

    print('ISI alignments', coverage(amrs, node_alignments))

    for amr in tqdm(amrs, file=sys.stdout):
        multiple_subgraph_alignments[amr.id] = []
        multiple_span_alignments[amr.id] = []

        if amr.id in node_alignments:
            aligns = node_alignments[amr.id]
            new_alignments = merge_spans(amr, aligns, amr.spans)
            subgraph_fuzzy_align(amr, new_alignments)
            if ENGLISH: subgraph_exact_align_english(amr, new_alignments)
            for align in new_alignments:
                if not align: continue
                new_aligns = separate_components(amr, align)
                if len(new_aligns)==1:
                    new_align = new_aligns[0]
                    new_align.type = 'subgraph'
                    add_alignment(amr, subgraph_alignments, new_align)
                    if VERBOSE:
                        print(new_align.readable(amr))
                else:
                    multiple_subgraph_alignments[amr.id].append(new_aligns)
            for n in amr.nodes:
                n_aligns = [a for a in subgraph_alignments[amr.id] if n in a.nodes]
                if len(n_aligns)>1:
                    for n_align in n_aligns:
                        remove_alignment(amr, subgraph_alignments, n_align)
                    multiple_span_alignments[amr.id].append(n_aligns)
    print('process ISI alignments',coverage(amrs, subgraph_alignments))
    subgraph_model.update_parameters(amrs, subgraph_alignments)

    # resolve alignments with multiple components
    resolve_inconsistent_alignments(amrs, subgraph_alignments, multiple_subgraph_alignments, subgraph_model)
    for amr in amrs:
        for n in amr.nodes:
            n_aligns = [a for a in subgraph_alignments[amr.id] if n in a.nodes]
            if len(n_aligns) > 1:
                for n_align in n_aligns:
                    remove_alignment(amr, subgraph_alignments, n_align)
                multiple_span_alignments[amr.id].append(n_aligns)
    print('resolve multiple subgraphs',coverage(amrs, subgraph_alignments))
    subgraph_model.update_parameters(amrs, subgraph_alignments)

    # resolve alignments with multiple spans
    for amr in amrs:
        align_groups = multiple_span_alignments[amr.id]
        for align_group in align_groups[:]:
            if len(align_group) == 2:
                align1 = align_group[0]
                align2 = align_group[1]
                if (len(align1.nodes) >1 or len(align2.nodes) >1) and len(set(align1.nodes) & set(align2.nodes)) == 1:
                    n = list(set(align1.nodes) & set(align2.nodes))[0]
                    if sum(1 for s,r,t in amr.edges if n in [s,t] and s in align1.nodes and t in align1.nodes)>1:
                        continue
                    if sum(1 for s,r,t in amr.edges if n in [s,t] and s in align2.nodes and t in align2.nodes)>1:
                        continue
                    if len(align1.nodes) > len(align2.nodes):
                        align2.nodes.remove(n)
                    else:
                        align1.nodes.remove(n)
                    if align1:
                        add_alignment(amr, subgraph_alignments, align1)
                    if align2:
                        add_alignment(amr, subgraph_alignments, align2)
                    multiple_span_alignments[amr.id].remove(align_group)

    resolve_inconsistent_alignments(amrs, subgraph_alignments, multiple_span_alignments, subgraph_model)
    print('resolve multiple spans', coverage(amrs, subgraph_alignments))
    subgraph_model.update_parameters(amrs, subgraph_alignments)

    if IGNORE_DUPLICATES:
        for amr in amrs:
            duplicates = [n for n in amr.nodes if len([n2 for n2 in amr.nodes if amr.nodes[n2]==amr.nodes[n]])>1]
            for align in subgraph_alignments[amr.id]:
                for n in align.nodes[:]:
                    if n in duplicates:
                        align.nodes.remove(n)
                postprocess_subgraph(amr, subgraph_alignments, align)
                if ENGLISH: postprocess_subgraph_english(amr, subgraph_alignments, align)
                clean_subgraph(amr, subgraph_alignments, align)
        print('removed duplicates', coverage(amrs, subgraph_alignments))

    subgraph_alignments = align_all(subgraph_model, amrs, subgraph_alignments)
    print('align all', coverage(amrs, subgraph_alignments))
    for amr in amrs:
        amr.alignments = subgraph_alignments[amr.id]
    Display.style(amrs[:100], amr_file.replace('.txt', '') + '.subgraphs.html')

    amrs_dict = {}
    for amr in amrs:
        amrs_dict[amr.id] = amr

    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.json'
    print(f'Writing subgraph alignments to: {align_file}')
    write_to_json(align_file, subgraph_alignments)



if __name__ == '__main__':
    main()
