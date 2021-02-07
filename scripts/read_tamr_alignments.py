import os

from amr_utils.alignments import AMR_Alignment, write_to_json
from amr_utils.amr_readers import JAMR_AMR_Reader


def main():
    dir = '../data/tamr'
    szubert_amrs = '../data/szubert/szubert_amrs.txt'
    output = '../data/szubert/szubert_amrs.tamr.subgraph_alignments.json'

    file2 = '../data/tamr/ldc_train_2017.txt'

    reader = JAMR_AMR_Reader()
    amrs = reader.load(szubert_amrs, remove_wiki=True)
    amrs2 = reader.load(file2, remove_wiki=True)

    alignments = {}
    for filename in os.listdir(dir):
        if filename.endswith(".tamr_alignment"):
            file = os.path.join(dir, filename)
            amr_id = ''
            with open(file) as f:
                for line in f:
                    if line.startswith('# ::alignments'):
                        aligns = line[len('# ::alignments '):].split()
                        aligns = [s.split('|') for s in aligns if '|' in s]
                        aligns = [(a[0], a[1].split('+')) for a in aligns]
                        for span, nodes in aligns:
                            start = int(span.split('-')[0])
                            end = int(span.split('-')[1])
                            span = [t for t in range(start, end)]
                            align = AMR_Alignment(type='subgraph', tokens=span, nodes=nodes)
                            alignments[amr_id].append(align)

                    elif line.strip():
                        amr_id = line.strip()
                        alignments[amr_id] = []

    amrs2 = {amr.id:amr for amr in amrs2}
    amrs = [amr for amr in amrs if amr.id in alignments and amr.id in amrs2]
    amrs3 = []
    for amr in amrs[:]:
        amr2 = amrs2[amr.id]
        nodes = {amr.nodes[n] for n in amr.nodes}
        nodes2 = {amr2.nodes[n] for n in amr2.nodes}
        edges = {(amr.nodes[s],r,amr.nodes[t]) for s,r,t in amr.edges}
        edges2 = {(amr2.nodes[s], r, amr2.nodes[t]) for s, r, t in amr2.edges}
        if nodes==nodes2 and edges==edges2:
            amrs3.append(amr)

    amr_ids = [amr.id for amr in amrs]
    alignments = {amr_id:alignments[amr_id] for amr_id in alignments if amr_id in amr_ids}
    for amr in amrs:
        node_map = {}
        nodes = [n for align in alignments[amr.id] for n in align.nodes]
        nodes = [n for n in sorted(nodes, key=lambda x:(len(x),x))]
        for n in nodes:
            prefix = '.'.join(i for i in n.split('.')[:-1])
            last = int(n.split('.')[-1])
            if prefix:
                if prefix not in node_map:
                    new_prefix = '.'.join(str(int(i) + 1) for i in n.split('.')[:-1])
                    if new_prefix not in amr.nodes:
                        continue
                    node_map[prefix] = new_prefix
                new_n = node_map[prefix] + '.' + str(last + 1)
            else:
                new_n = str(last + 1)
            if new_n in amr.nodes:
                node_map[n] = new_n
        nodes = [n for align in alignments[amr.id] for n in align.nodes if n not in node_map]
        nodes = [n for n in sorted(nodes, key=lambda x: (len(x),x))]
        for n in nodes:
            prefix = '.'.join(i for i in n.split('.')[:-1])
            if prefix not in node_map:
                new_prefix = '.'.join(str(int(i) + 1) for i in n.split('.')[:-1])
                if new_prefix in amr.nodes:
                    node_map[prefix] = new_prefix
                else:
                    del alignments[amr.id]
                    break
            candidates = [t for s,r,t in amr.edges if s==node_map[prefix]]
            candidates = [t for t in candidates if t not in node_map.values()]
            candidates = [t for t in sorted(candidates)]
            if not candidates:
                del alignments[amr.id]
                break
            new_n = candidates[0]
            node_map[n] = new_n
        if amr.id in alignments:
            for align in alignments[amr.id]:
                align.nodes = [node_map[n] for n in align.nodes]
                align.amr = amr
            for t,tok in enumerate(amr.tokens):
                align = amr.get_alignment(alignments, token_id=t)
                if not align:
                    align = AMR_Alignment(type='subgraph', tokens=[t], nodes=[], amr=amr)
                    alignments[amr.id].append(align)
            alignments[amr.id] = [align for align in sorted(alignments[amr.id], key=lambda a:a.tokens[0])]

    write_to_json(output, alignments)


if __name__=='__main__':
    main()