import sys
import random

from amr_utils.amr_readers import AMR_Reader
from amr_utils.style import HTML_AMR

from display import Display


class ID_Display:

    @staticmethod
    def style(amrs, outfile):
        output = HTML_AMR.style(amrs[:5000],
                                assign_node_desc=ID_Display.node_desc,
                                assign_token_desc=ID_Display.token_desc,
                                assign_edge_desc=ID_Display.edge_desc,)

        with open(outfile, 'w+', encoding='utf8') as f:
            f.write(output)

    @staticmethod
    def node_desc(amr, n):
        node_labels = get_node_labels(amr)
        return node_labels[n] + ' : ' + Display.node_desc(amr, n)

    @staticmethod
    def edge_desc(amr, e):
        edge_labels = get_edge_labels(amr)
        return edge_labels[e]

    @staticmethod
    def token_desc(amr, tok):
        desc1 = str(tok)
        desc2 = Display.token_desc(amr, tok)
        return f'{desc1} : {desc2}'

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


def load_szubert_data(amr_file):

    # Szubert data
    reader = AMR_Reader()
    amrs1 = reader.load(amr_file, remove_wiki=True)
    amr_ids = [amr.id for amr in amrs1]
    for amr_id in amr_ids:
        if amr_ids.count(amr_id)>1:
            print('Repeated:', amr_id)

    # LDC data
    amrs2 = []
    amrs2 += reader.load('data/ldc_train.txt', remove_wiki=True)
    amrs2 += reader.load('data/ldc_dev.txt', remove_wiki=True)
    amrs2 += reader.load('data/ldc_test.txt', remove_wiki=True)
    amrs = [amr for amr in amrs2 if amr.id in amr_ids]
    ldc_ids = [amr.id for amr in amrs]

    # Little Prince data
    amrs3 = reader.load('data/little_prince.txt', remove_wiki=True)
    little_prince_ids = [amr.id for amr in amrs3 if amr.id in amr_ids]
    amrs += [amr for amr in amrs3 if amr.id in little_prince_ids]

    # other data
    other_ids = [amr_id for amr_id in amr_ids if amr_id not in ldc_ids and amr_id not in little_prince_ids]
    amrs += [amr for amr in amrs1 if amr.id in other_ids]

    print('Missing:', ' '.join(i for i in other_ids))
    print(len(amrs), '/', len(amrs1), 'AMRs printed')

    return amrs




def main():
    amr_file = sys.argv[1]
    alignment_file = sys.argv[2]
    relation_alignment_file = sys.argv[3]

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)

    subgraph_alignments = reader.load_alignments_from_json(alignment_file, amrs)
    relation_alignments = reader.load_alignments_from_json(relation_alignment_file, amrs)

    amrs = [amr for amr in amrs if amr.id in subgraph_alignments]
    for amr in amrs:
        amr.alignments = subgraph_alignments[amr.id]
    # random.shuffle(amrs)
    # amrs = amrs[:100]
    # print('Sampling AMRs:')
    # for amr in amrs:
    #     print(amr.id)

    output_file = amr_file.replace('.txt','.gold.txt')
    with open(output_file, 'w+', encoding='utf8') as f:
        for amr in amrs:
            f.write(amr.jamr_string())

    output_file2 = output_file.replace('.txt','.html')
    ID_Display.style(amrs, output_file2)

    output_file3 = output_file.replace('.gold.txt','.gold_alignments.tsv')
    with open(output_file3, 'w+', encoding='utf8') as f:
        for amr in amrs:
            f.write('\t'.join(['amr',str(amr.id)])+'\n')
            reentrancies = []
            for n in amr.nodes:
                parents = [(s,r,t) for s,r,t in amr.edges if t==n]
                if len(parents)>1:
                    reentrancies.extend(parents)
            node_labels = get_node_labels(amr)
            edge_labels = get_edge_labels(amr)
            f.write('\t'.join(['tokens']+[f'{i}={token}' for i,token in enumerate(amr.tokens)])+'\n')
            for n in amr.nodes:
                nalign = amr.get_alignment(subgraph_alignments, node_id=n)
                if nalign:
                    token_ids = nalign.tokens
                    token_ids = ','.join(str(t) for t in token_ids)
                    f.write('\t'.join(['node',node_labels[n], amr.nodes[n], token_ids])+'\n')
                else:
                    f.write('\t'.join(['node', node_labels[n], amr.nodes[n], '']) + '\n')
            for s,r,t in amr.edges:
                ealign = amr.get_alignment(relation_alignments, edge=(s,r,t))
                if ealign:
                    token_ids = ealign.tokens
                    token_ids = ','.join(str(t) for t in token_ids)
                    f.write('\t'.join(['edge', edge_labels[(s,r,t)], f'{amr.nodes[s]} {r} {amr.nodes[t]}', token_ids])+'\n')
                else:
                    f.write('\t'.join(['edge', edge_labels[(s,r,t)], f'{amr.nodes[s]} {r} {amr.nodes[t]}', '']) + '\n')
            for s,r,t in reentrancies:
                f.write('\t'.join(['reentrancy', edge_labels[(s,r,t)], f'{amr.nodes[s]} {r} {amr.nodes[t]}']) + '\n')







if __name__=='__main__':
    main()

