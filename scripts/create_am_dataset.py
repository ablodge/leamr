import sys

from amr_utils.alignments import load_from_json
from amr_utils.amr_readers import JAMR_AMR_Reader


def main():
    amr_file = sys.argv[1]
    align_file = sys.argv[2]
    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)

    subgraph_alignments = load_from_json(align_file, amrs)
    for amr_id in subgraph_alignments:
        if any(align.type!='subgraph' for align in subgraph_alignments[amr_id]):
            subgraph_alignments[amr_id] = [align for align in subgraph_alignments[amr_id] if align.type=='subgraph']

    new_amrs = []
    for amr in amrs:
        reentrancies = []
        for n in amr.nodes:
            parents = [s for s,r,t in amr.edges if n==t]
            if len(parents)>1:
                reentrancies.append(n)
        unaligned = [n for n in amr.nodes]
        for align in subgraph_alignments[amr.id]:
            for n in align.nodes:
                if n in unaligned:
                    unaligned.remove(n)
        names = [n for n in amr.nodes if amr.nodes[n]=='name']
        if amr.tokens[0] == '[':
            continue
        if not reentrancies and not unaligned and len(names)<10:
            new_amrs.append(amr)

    print(f'available amrs: {len(new_amrs)} / {len(amrs)} = {len(new_amrs)/len(amrs)}')
    TRAIN_SIZE = 10000
    train_amrs = new_amrs[:TRAIN_SIZE]
    dev_amrs = new_amrs[TRAIN_SIZE:TRAIN_SIZE+1000]
    test_amrs = new_amrs[TRAIN_SIZE+1000:TRAIN_SIZE+2000]

    new_name = amr_file.replace('.txt', '.small.txt')
    with open(new_name, 'w+', encoding='utf8') as f:
        for amr in train_amrs:
            f.write(f'# ::id {amr.id}\n')
            f.write(f'# ::snt {" ".join(amr.tokens)}\n')
            f.write(amr.graph_string())

    new_name = amr_file.replace('.txt', '.small.txt').replace('ldc_train','ldc_dev')
    with open(new_name, 'w+', encoding='utf8') as f:
        for amr in dev_amrs:
            f.write(f'# ::id {amr.id}\n')
            f.write(f'# ::snt {" ".join(amr.tokens)}\n')
            f.write(amr.graph_string())

    new_name = amr_file.replace('.txt', '.small.txt').replace('ldc_train','ldc_test')
    with open(new_name, 'w+', encoding='utf8') as f:
        for amr in test_amrs:
            f.write(f'# ::id {amr.id}\n')
            f.write(f'# ::snt {" ".join(amr.tokens)}\n')
            f.write(amr.graph_string())

if __name__=='__main__':
    main()