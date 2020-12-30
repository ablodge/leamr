import csv
import sys

from amr_utils.alignments import AMR_Alignment, write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader

from display import Display


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

    amr_reader = JAMR_AMR_Reader()
    amrs = amr_reader.load(amr_file, remove_wiki=True)
    amrs = {amr.id:amr for amr in amrs}

    subgraph_alignments = {amr_id:[] for amr_id in amrs}
    amr = None
    node_labels = {}
    with open(hand_alignments_file) as f:
        hand_alignments = csv.reader(f, delimiter="\t")
        for row in hand_alignments:
            if row[0]=='amr':
                amr_id = row[1]
                amr = amrs[amr_id]
                node_labels = get_node_labels(amr)
                node_labels = {v:k for k,v in node_labels.items()}
                # edge_labels = get_edge_labels(amr)
                # edge_labels = {v: k for k, v in edge_labels.items()}
            elif row[0]=='node':
                if row[3].startswith('*'): continue
                if not row[3]:
                    raise Exception('Missing Annotation:', amr_id)
                node_id = row[1]
                n = node_labels[node_id]
                token_ids = [int(t) for t in row[3].split(',')]
                align = amr.get_alignment(subgraph_alignments, token_id=token_ids[0])
                if align:
                    align.nodes.append(n)
                else:
                    new_align = AMR_Alignment(type='subgraph',tokens=token_ids,nodes=[n], amr=amr)
                    subgraph_alignments[amr.id].append(new_align)
    for amr_id in amrs:
        alignments = subgraph_alignments[amr_id]
        amr = amrs[amr_id]
        for t in range(len(amr.tokens)):
            if not any(t in align.tokens for align in alignments):
                alignments.append(AMR_Alignment(type='subgraph',tokens=[t], amr=amr))
        alignments = [align for align in sorted(alignments, key=lambda x:x.tokens[0])]
        subgraph_alignments[amr_id] = alignments

    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.gold.json'
    print(f'Writing subgraph alignments to: {align_file}')
    write_to_json(align_file, subgraph_alignments)

    amrs = [amr for amr_id,amr in amrs.items()]
    for amr in amrs:
        amr.alignments = subgraph_alignments[amr.id]
    Display.style(amrs, amr_file.replace('.txt', '') + f'.hand_alignments.html')




if __name__=='__main__':
    main()