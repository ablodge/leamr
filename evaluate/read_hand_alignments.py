import csv
import sys

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr_readers import AMR_Reader

from rule_based.subgraph_rules import is_subgraph


def clean_alignments(amr, alignments, duplicate_alignments, spans, mode='subgraph'):
    aligns = []
    for span in spans:
        align = amr.get_alignment(alignments, token_id=span[0])
        if align:
            aligns.append(align)
        else:
            aligns.append(AMR_Alignment(type=mode, tokens=span, amr=amr))
    alignments[amr.id] = aligns + duplicate_alignments


def get_node_labels(amr):
    node_labels = {}
    taken = set()
    for n in amr.nodes:
        letter = amr.nodes[n].lower()[0]
        if not letter.isalpha() or amr.nodes[n] in ['imperative', 'expressive', 'interrogative']:
            letter = 'x'
        if letter in taken:
            i = 2
            while f'{letter}{i}' in taken:
                i+=1
            letter = f'{letter}{i}'
        node_labels[n] = letter
        taken.add(letter)
    return node_labels

def get_edge_labels(amr):
    node_labels = get_node_labels(amr)
    return {(s,r,t):f'{node_labels[s]}.{node_labels[t]}' for s,r,t in amr.edges}

def main():
    amr_file = sys.argv[1]
    hand_alignments_file = sys.argv[2]

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    amrs = {amr.id:amr for amr in amrs}

    subgraph_alignments = {}
    relation_alignments = {}
    reentrancy_alignments = {}
    all_spans = {amr_id:set() for amr_id in amrs}

    amr = None
    node_labels = {}
    with open(hand_alignments_file) as f:
        hand_alignments = csv.reader(f, delimiter="\t")
        for row in hand_alignments:
            if row[0]=='amr':
                amr_id = row[1]
                subgraph_alignments[amr_id] = []
                relation_alignments[amr_id] = []
                reentrancy_alignments[amr_id] = []
                amr = amrs[amr_id]
                taken = set()
                node_labels = get_node_labels(amr)
                node_labels = {v:k for k,v in node_labels.items()}
                edge_labels = get_edge_labels(amr)
                edge_labels = {v: k for k, v in edge_labels.items()}
            elif row[0]=='node':
                type = 'subgraph'
                if row[3].startswith('*'):
                    type = 'dupl-subgraph'
                    row[3] = row[3].replace('*','')
                if not row[3]:
                    raise Exception('Missing Annotation:', amr_id)
                node_id = row[1]
                n = node_labels[node_id]
                token_ids = [int(t) for t in row[3].split(',')]
                if any(t>=len(amr.tokens) for t in token_ids):
                    raise Exception('Bad Annotation:', amr_id)
                if tuple(token_ids) not in all_spans[amr_id] and any(t in taken for t in token_ids):
                    raise Exception('Bad Span Annotation', amr_id)
                all_spans[amr_id].add(tuple(token_ids))
                taken.update(token_ids)
                align = amr.get_alignment(subgraph_alignments, token_id=token_ids[0])
                if align and align.type==type:
                    align.nodes.append(n)
                else:
                    new_align = AMR_Alignment(type=type,tokens=token_ids,nodes=[n], amr=amr)
                    subgraph_alignments[amr.id].append(new_align)
            elif row[0]=='edge':
                type = 'relation'
                if row[3].startswith('*'):
                    row[3] = row[3].replace('*','')
                if not row[3]:
                    raise Exception('Missing Annotation:', amr_id)
                edge_id = row[1]
                e = edge_labels[edge_id]
                token_ids = [int(t) for t in row[3].split(',')]
                if any(t>=len(amr.tokens) for t in token_ids):
                    raise Exception('Bad Annotation:', amr_id)
                if tuple(token_ids) not in all_spans[amr_id] and any(t in taken for t in token_ids):
                    raise Exception('Bad Span Annotation', amr_id, token_ids)
                all_spans[amr_id].add(tuple(token_ids))
                taken.update(token_ids)
                align = amr.get_alignment(relation_alignments, token_id=token_ids[0])
                if align and align.type==type:
                    align.edges.append(e)
                else:
                    new_align = AMR_Alignment(type=type,tokens=token_ids,edges=[e], amr=amr)
                    relation_alignments[amr.id].append(new_align)
            elif row[0]=='reentrancy':
                if not row[3]:
                    raise Exception('Missing Annotation:', amr_id)
                edge_id = row[1]
                e = edge_labels[edge_id]
                if row[3].startswith('*'):
                    row[3] = row[3].replace('*','')
                if row[3]=='_':
                    token_ids = amr.get_alignment(relation_alignments, edge=e).tokens
                else:
                    token_ids = [int(t) for t in row[3].split(',')]
                tag = row[4]
                if row[3]=='_':
                    tag='primary'
                if not tag:
                    raise Exception('Missing reentrancy tag:', amr.id)
                type = f'reentrancy:{tag}'
                if any(t>=len(amr.tokens) for t in token_ids):
                    raise Exception('Bad Annotation:', amr_id)
                if tuple(token_ids) not in all_spans[amr_id] and any(t in taken for t in token_ids):
                    raise Exception('Bad Span Annotation', amr_id, token_ids)
                all_spans[amr_id].add(tuple(token_ids))
                taken.update(token_ids)
                new_align = AMR_Alignment(type=type,tokens=token_ids,edges=[e], amr=amr)
                reentrancy_alignments[amr.id].append(new_align)
    for amr_id in subgraph_alignments:
        amr = amrs[amr_id]
        for t in range(len(amr.tokens)):
            if not any(t in span for span in all_spans[amr_id]):
                all_spans[amr_id].add((t,))
        spans = [list(span) for span in sorted(all_spans[amr_id],key = lambda x:x[0])]

        for align in subgraph_alignments[amr_id]:
            if align.nodes and not is_subgraph(amr, align.nodes):
                print('Possible Bad align:', amr.id, align.tokens, ' '.join(amr.tokens[t] for t in align.tokens), file=sys.stderr)
        for align in relation_alignments[amr_id]:
            subgraph_aligns = [a for a in subgraph_alignments[amr.id] if a.tokens==align.tokens]
            for s,r,t in align.edges:
                if subgraph_aligns and not any(s in a.nodes or t in a.nodes or not a.nodes for a in subgraph_aligns):
                    raise Exception('Bad Relation align:', amr.id, align.tokens, s, r, t)
        dupl_sub_aligns = [align for align in subgraph_alignments[amr_id] if align.type.startswith('dupl')]
        subgraph_alignments[amr_id] = [align for align in subgraph_alignments[amr_id] if not align.type.startswith('dupl')]
        # dupl_rel_aligns = [align for align in relation_alignments[amr_id] if align.type.startswith('dupl')]
        # relation_alignments[amr_id] = [align for align in relation_alignments[amr_id] if not align.type.startswith('dupl')]
        clean_alignments(amr, subgraph_alignments, dupl_sub_aligns, spans)
        clean_alignments(amr, relation_alignments, [], spans, mode='relations')
        for t,_ in enumerate(amr.tokens):
            count = [span for span in spans if t in span]
            if len(count)!=1:
                raise Exception('Bad Span:', amr.id, count)

    # amr_file = amr_file.replace('.txt', '.jakob')
    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.gold.json'
    print(f'Writing subgraph alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, subgraph_alignments)

    align_file = amr_file.replace('.txt', '') + f'.relation_alignments.gold.json'
    print(f'Writing relation alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, relation_alignments)

    align_file = amr_file.replace('.txt', '') + f'.reentrancy_alignments.gold.json'
    print(f'Writing reentrancy alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, reentrancy_alignments)

    # amrs = [amr for amr_id,amr in amrs.items() if amr_id in subgraph_alignments]
    # for amr in amrs:
    #     amr.alignments = subgraph_alignments[amr.id]
    # Display.style(amrs, amr_file.replace('.txt', '') + f'.hand_alignments.html')




if __name__=='__main__':
    main()