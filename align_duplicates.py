import sys

from amr_utils.alignments import AMR_Alignment, load_from_json, write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader
from tqdm import tqdm

from display import Display
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data
from rule_based.subgraph_rules import postprocess_subgraph, clean_subgraph

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
            postprocess_subgraph(amr, alignments, align, english=ENGLISH)
            clean_subgraph(amr, alignments, align, english=ENGLISH)
            return
    raise Exception('Alignment not found!')


def resolve_duplicate_alignments(amr, subgraph_alignments, duplicate_alignments, candidate_spans, candidate_nodes, model):
    scores = {}
    aligns = {}
    all_align = {amr.id: subgraph_alignments[amr.id] + duplicate_alignments[amr.id]}

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
    if len(candidate_nodes)>len(candidate_spans):
        for s,r,t in amr.edges:
            if amr.nodes[s] in ['include-91','same-01','instead-of-91','resemble-01','and','or'] and r.endswith('1') and t in candidate_nodes:
                for s2, r2, t2 in amr.edges:
                    if s2==s and t2!=t and t2 in candidate_nodes:
                        ignore_list.append(candidate_nodes.index(t2))
            elif s in candidate_nodes and r.endswith('1-of') and amr.nodes[t] in ['include-91','same-01','instead-of-91','and','or']:
                for s2, r2, t2 in amr.edges:
                    if s2==t and t2!=s and t2 in candidate_nodes:
                        ignore_list.append(candidate_nodes.index(t2))

    for j,n in enumerate(candidate_nodes):
        if amr.get_alignment(all_align, node_id=n):
            taken_nodes.add(j)

    for i,j in best_scores:
        if i in taken_spans or j in taken_nodes: continue
        if j in ignore_list: continue
        align = aligns[(i, j)]
        add_alignment(amr, subgraph_alignments, align)
        taken_spans.add(i)
        taken_nodes.add(j)

    # if len(taken_spans) < len(candidate_spans):
    #     for i, j in best_scores:
    #         if i in taken_spans and j in taken_nodes: continue
    #         if i not in taken_spans:
    #             align = aligns[(i, j)].copy()
    #             align.type = 'dupl-span'
    #             duplicate_alignments[amr.id].append(align)
    #             taken_spans.add(i)
    if len(taken_nodes) < len(candidate_nodes):
        for i, j in best_scores:
            if i in taken_spans and j in taken_nodes: continue
            if j not in taken_nodes:
                align = aligns[(i, j)].copy()
                if len(align.nodes)==1 and amr.nodes[align.nodes[0]] in ['thing','and','-','person']:
                    continue
                align.type = 'dupl-subgraph'
                replaced_align = amr.get_alignment(subgraph_alignments, token_id=candidate_spans[i][0])
                align.nodes = [n for n in align.nodes if n not in replaced_align.nodes]
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

def align_duplicates(amrs, subgraph_model, subgraph_alignments):
    duplicate_alignments = {}

    for amr in tqdm(amrs, file=sys.stdout):
        duplicate_alignments[amr.id] = []

        # find duplicate nodes
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

        # align duplicate nodes
        duplicates_todo = {node_label for node_label in duplicate_nodes}
        while duplicates_todo:
            rank = {}
            candidate_tokens = {node_label:set() for node_label in duplicates_todo}
            readable = []
            for node_label, candidate_nodes in duplicate_nodes.items():
                if node_label not in duplicates_todo: continue
                for n in candidate_nodes:
                    best_align, best_score = subgraph_model.align(amr, subgraph_alignments, n)
                    if best_align is None:
                        continue
                    candidate_tokens[node_label].add(' '.join(amr.lemmas[t].lower() for t in best_align.tokens))
                    rank[(node_label,n)] = best_score
                    # readable.append((best_score,
                    #                  best_align.tokens,
                    #                  node_label,
                    #                  subgraph_model.readable_logp(amr, subgraph_alignments, best_align)))
                exact_match = node_label.replace('-',' ')
                if exact_match[-1].isdigit():
                    exact_match = ' '.join(exact_match.split()[:-1])
                candidate_tokens[node_label].add(exact_match)
            best_node_label, n = max(rank.keys(), key=lambda x:rank[x])
            candidate_spans = [span for span in amr.spans if ' '.join(amr.lemmas[t].lower() for t in span) in candidate_tokens[best_node_label]]
            candidate_nodes = duplicate_nodes[best_node_label]
            resolve_duplicate_alignments(amr, subgraph_alignments,
                                         duplicate_alignments,
                                         candidate_spans, candidate_nodes, subgraph_model)
            # readable = [r for r in sorted(readable, key=lambda x:x[0], reverse=True)]
            duplicates_todo.remove(best_node_label)
    print('align duplicates', coverage(amrs, subgraph_alignments))
    subgraph_model.update_parameters(amrs, subgraph_alignments)

    for k in subgraph_alignments:
        if k not in duplicate_alignments:
            duplicate_alignments[k] = []

    all_alignments = {}
    for k in subgraph_alignments:
        all_alignments[k] = subgraph_alignments[k]+duplicate_alignments[k]

    all_alignments = subgraph_model.align_all(amrs, all_alignments)
    for k in subgraph_alignments:
        subgraph_alignments[k] = [align for align in all_alignments[k] if align.type=='subgraph']

    print('align all', coverage(amrs, subgraph_alignments), '+', coverage(amrs, duplicate_alignments))
    for amr in amrs:
        amr.alignments = all_alignments[amr.id]

    return subgraph_alignments, duplicate_alignments



def main():
    amr_file = sys.argv[1]
    align_file = sys.argv[1].replace('.txt', '') + '.subgraph_alignments.tmp1.json'

    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)
    if TRAIN_MAX:
        amrs = amrs[:TRAIN_MAX]

    add_nlp_data(amrs, amr_file)

    subgraph_alignments = load_from_json(align_file, amrs)

    subgraph_model = Subgraph_Model(amrs, ignore_duplicates=False)
    subgraph_model.update_parameters(amrs, subgraph_alignments)

    subgraph_alignments, duplicate_alignments = align_duplicates(amrs, subgraph_model, subgraph_alignments)

    display_file = amr_file.replace('.txt', '') + '.duplicates.html'
    print(f'Creating alignments display file: {display_file}')
    Display.style(amrs[:100], display_file)

    amrs_dict = {}
    for amr in amrs:
        amrs_dict[amr.id] = amr

    for amr_id in subgraph_alignments:
        for align in subgraph_alignments[amr_id]:
            if align.type!='subgraph':
                raise Exception('Incorrect alignment format!')

    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.json'
    print(f'Writing subgraph alignments to: {align_file}')
    write_to_json(align_file, subgraph_alignments)

    align_file = amr_file.replace('.txt', '') + f'.duplicate_alignments.json'
    print(f'Writing duplicate alignments to: {align_file}')
    write_to_json(align_file, duplicate_alignments)


if __name__ == '__main__':
    main()
