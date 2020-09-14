import sys

from amr_utils.alignments import AMR_Alignment, load_from_json, write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader
from amr_utils.graph_utils import breadth_first_edges
from tqdm import tqdm

from display import Display
from models.subgraph_model import Subgraph_Model
from models.utils import align_all
from nlp_data import add_nlp_data
from rule_based.subgraph_rules import postprocess_subgraph, postprocess_subgraph_english, clean_subgraph

ENGLISH = True

TRAIN_MAX = None
# TRAIN_MAX = 100

def add_alignment(amr, alignments, align):
    for j,old_align in enumerate(alignments[amr.id]):
        if old_align.tokens == align.tokens:
            alignments[amr.id][j] = align
            for n in old_align.nodes:
                if n not in align.nodes:
                    align.nodes.append(n)
            postprocess_subgraph(amr, alignments, align)
            if ENGLISH: postprocess_subgraph_english(amr, alignments, align)
            clean_subgraph(amr, alignments, align)
            return
    raise Exception('Alignment not found!')


def resolve_duplicate_alignments(amr, subgraph_alignments, duplicate_alignments, candidate_spans, candidate_nodes, model):
    scores = {}
    aligns = {}

    for i,span in enumerate(candidate_spans):
        for j,n in enumerate(candidate_nodes):
            replaced_align = amr.get_alignment(subgraph_alignments, token_id=span[0])
            new_align = AMR_Alignment(type='subgraph', tokens=span, nodes=[n]+replaced_align.nodes, amr=amr)
            aligns[(i,j)] = new_align
            scores[(i,j)] = model.logp(amr, subgraph_alignments, new_align) - model.logp(amr, subgraph_alignments, replaced_align)
    best_scores = sorted(scores, key=lambda x:scores[x], reverse=True)
    taken_spans = set()
    taken_nodes = set()
    ignore_list = []
    for j,n in enumerate(candidate_nodes):
        for s,r,t in amr.edges:
            if t == n and amr.nodes[s] in ['include-91','same-01','instead-of-91']:
                if any(t2==s and s2 in candidate_nodes for s2,r2,t2 in amr.edges):
                    ignore_list.append(j)
                elif any(s2==s and t2!=t and t2 in candidate_nodes for s2,r2,t2 in amr.edges):
                    ignore_list.append(j)

    for i,j in best_scores:
        if i in taken_spans or j in taken_nodes: continue
        if j in ignore_list: continue
        align = aligns[(i, j)]
        add_alignment(amr, subgraph_alignments, align)
        taken_spans.add(i)
        taken_nodes.add(j)

    if len(taken_spans) < len(candidate_spans):
        for i, j in best_scores:
            if i in taken_spans and j in taken_nodes: continue
            if i not in taken_spans:
                align = aligns[(i, j)].copy()
                align.type = 'dupl-span'
                duplicate_alignments[amr.id].append(align)
                taken_spans.add(i)
    if len(taken_nodes) < len(candidate_nodes):
        for i, j in best_scores:
            if i in taken_spans and j in taken_nodes: continue
            if j not in taken_nodes:
                align = aligns[(i, j)].copy()
                if len(align.nodes)==1 and amr.nodes[align.nodes[0]] in ['thing','and','-','person']:
                    continue
                align.type = 'dupl-subgraph'
                duplicate_alignments[amr.id].append(align)
                taken_nodes.add(j)


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
    align_file = sys.argv[1].replace('.txt', '') + '.subgraph_alignments.no-pretrain0.json'

    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)
    if TRAIN_MAX:
        amrs = amrs[:TRAIN_MAX]

    add_nlp_data(amrs, amr_file)

    subgraph_alignments = load_from_json(align_file, amrs)
    duplicate_alignments = {}

    subgraph_model = Subgraph_Model(amrs, ignore_duplicates=False)
    subgraph_model.update_parameters(amrs, subgraph_alignments)

    # lower the importance of distance to better handle reentrancy
    dist_stdev = subgraph_model.distance_model.distance_stdev
    dist_mean = subgraph_model.distance_model.distance_mean
    subgraph_model.distance_model.update_parameters(dist_mean, 2*dist_stdev)

    for amr in tqdm(amrs, file=sys.stdout):
        duplicate_alignments[amr.id] = []
        duplicate_nodes = {}
        unaligned = subgraph_model.get_unaligned(amr, subgraph_alignments)
        for n in amr.nodes:
            if n not in unaligned:
                continue
            node_label = amr.nodes[n]
            if node_label not in duplicate_nodes:
                duplicate_nodes[node_label] = []
            duplicate_nodes[amr.nodes[n]].append(n)
        duplicate_nodes = {node_label:duplicate_nodes[node_label] for node_label in duplicate_nodes if len(duplicate_nodes[node_label])>1}
        # find_duplicate_subgraphs(amr, duplicate_nodes)
        candidate_scores = {}
        candidate_spans = {}
        for node_label, candidate_nodes in duplicate_nodes.items():
            candidate_tokens = set()
            score = float('-inf')
            for n in candidate_nodes:
                best_align, best_score = subgraph_model.align(amr, subgraph_alignments, n, unaligned)
                if best_align is None: continue
                candidate_tokens.add(' '.join(amr.lemmas[t] for t in best_align.tokens))
                if best_score > score:
                    score = best_score
            candidate_scores[node_label] = score
            candidate_spans[node_label] = [span for span in amr.spans if ' '.join(amr.lemmas[t] for t in span) in candidate_tokens]
        for node_label in sorted(candidate_scores.keys(), key=lambda x:candidate_scores[x], reverse=True):
            resolve_duplicate_alignments(amr, subgraph_alignments,
                                         duplicate_alignments,
                                         candidate_spans[node_label], duplicate_nodes[node_label], subgraph_model)
    print('resolve duplicates', coverage(amrs, subgraph_alignments))
    subgraph_model.update_parameters(amrs, subgraph_alignments)

    for k in subgraph_alignments:
        if k not in duplicate_alignments:
            duplicate_alignments[k] = []
    all_alignments = {k:subgraph_alignments[k]+duplicate_alignments[k] for k in subgraph_alignments}

    subgraph_alignments = align_all(subgraph_model, amrs, all_alignments)
    print('align all', coverage(amrs, subgraph_alignments))
    for amr in amrs:
        amr.alignments = subgraph_alignments[amr.id] + duplicate_alignments[amr.id]

    display_file = amr_file.replace('.txt', '') + '.duplicates.html'
    print(f'Creating alignments display file: {display_file}')
    Display.style(amrs[:100], display_file)

    amrs_dict = {}
    for amr in amrs:
        amrs_dict[amr.id] = amr

    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments2.json'
    print(f'Writing subgraph alignments to: {align_file}')
    write_to_json(align_file, subgraph_alignments)

    align_file = amr_file.replace('.txt', '') + f'.duplicate_alignments.json'
    print(f'Writing duplicate alignments to: {align_file}')
    write_to_json(align_file, duplicate_alignments)


if __name__ == '__main__':
    main()
