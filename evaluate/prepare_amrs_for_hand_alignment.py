import sys
import random

from amr_utils.amr_readers import LDC_AMR_Reader, JAMR_AMR_Reader
from amr_utils.style import HTML_AMR


class Display:

    @staticmethod
    def style(amrs, outfile):
        output = HTML_AMR.style(amrs[:5000],
                                assign_node_desc=Display.node_desc,
                                assign_token_desc=Display.token_desc,
                                assign_edge_desc=Display.edge_desc)

        with open(outfile, 'w+', encoding='utf8') as f:
            f.write(output)

    @staticmethod
    def node_desc(amr, n):
        node_labels = get_node_labels(amr)
        return node_labels[n]

    @staticmethod
    def edge_desc(amr, e):
        edge_labels = get_edge_labels(amr)
        return edge_labels[e]

    @staticmethod
    def token_desc(amr, tok):
        return str(tok)

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
    output_file = sys.argv[3]

    # Szubert data
    cr = LDC_AMR_Reader()
    amrs1 = cr.load(amr_file, remove_wiki=True)
    amr_ids = [amr.id for amr in amrs1]
    for amr_id in amr_ids:
        if amr_ids.count(amr_id)>1:
            print('Repeated:', amr_id)

    # LDC data
    cr = JAMR_AMR_Reader()
    amrs2 = []
    amrs2 += cr.load('data/ldc_train.txt', remove_wiki=True)
    amrs2 += cr.load('data/ldc_dev.txt', remove_wiki=True)
    amrs2 += cr.load('data/ldc_test.txt', remove_wiki=True)
    amrs = [amr for amr in amrs2 if amr.id in amr_ids]
    ldc_ids = [amr.id for amr in amrs]

    # Little Prince data
    cr = LDC_AMR_Reader()
    amrs3 = cr.load('data/little_prince.txt', remove_wiki=True)
    little_prince_ids = [amr.id for amr in amrs3 if amr.id in amr_ids]
    amrs += [amr for amr in amrs3 if amr.id in little_prince_ids]

    # other data
    other_ids = [amr_id for amr_id in amr_ids if amr_id not in ldc_ids and amr_id not in little_prince_ids]
    amrs += [amr for amr in amrs1 if amr.id in other_ids]

    print('Missing:', ' '.join(i for i in other_ids))
    print(len(amrs), '/', len(amrs1), 'AMRs printed')

    random.shuffle(amrs)
    amrs = amrs[:20]
    print('Sampling AMRs:')
    for amr in amrs:
        print(amr.id)




    output_file = output_file.replace('.txt','.html')
    Display.style(amrs, output_file)

    output_file = output_file.replace('.html','.hand_alignments.tsv')
    with open(output_file, 'w+', encoding='utf8') as f:
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
                f.write('\t'.join(['node',node_labels[n], amr.nodes[n]])+'\n')
            for s,r,t in amr.edges:
                f.write('\t'.join(['edge', edge_labels[(s,r,t)], f'{amr.nodes[s]} {r} {amr.nodes[t]}']) + '\n')
            for s,r,t in reentrancies:
                f.write('\t'.join(['reentrancy', edge_labels[(s,r,t)], f'{amr.nodes[s]} {r} {amr.nodes[t]}']) + '\n')







if __name__=='__main__':
    main()

