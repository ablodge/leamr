import sys

from amr_utils.alignments import write_to_json
from amr_utils.amr_readers import AMR_Reader


def main():
    amr_file_old = sys.argv[1]
    amr_file_new = sys.argv[2]
    output_file = sys.argv[3]

    reader = AMR_Reader()
    amrs_old = reader.load(amr_file_old)
    amrs_new = reader.load(amr_file_new, remove_wiki=True)


    bad_node_map = {}
    for amr1 in amrs_new:
        amr2 = next(a for a in amrs_old if a.id == amr1.id)
        for n in amr1.nodes:
            amr1.nodes[n] = amr1.nodes[n].replace('"','')
        for n in amr2.nodes:
            amr2.nodes[n] = amr2.nodes[n].replace('"','')
        bad_nodes = []
        for n in amr1.nodes:
            if n not in amr2.nodes or amr1.nodes[n] != amr2.nodes[n]:
                bad_nodes.append(n)
                continue
            neighborhood = {f'{amr1.nodes[e[0]]} {e[1]} {amr1.nodes[e[2]]}' for e in amr1.edges if n in e}
            neighborhood2 = {f'{amr2.nodes[e[0]]} {e[1]} {amr2.nodes[e[2]]}' for e in amr2.edges if n in e}
            if neighborhood!=neighborhood2:
                bad_nodes.append(n)
        if bad_nodes:
            bad_node_map[amr1.id] = {}
            for n in amr1.nodes:
                if n in bad_nodes:
                    new_n = [n2 for n2 in amr2.nodes
                             if amr2.nodes[n2]==amr1.nodes[n]
                             and not (n2 in amr1.nodes and amr1.nodes[n2]==amr1.nodes[n])]
                    neighborhood = {f'{amr1.nodes[e[0]]} {e[1]} {amr1.nodes[e[2]]}' for e in amr1.edges if n in e}
                    if len(new_n)>1:
                        neighborhood2 = {n2:[f'{amr2.nodes[e[0]]} {e[1]} {amr2.nodes[e[2]]}' for e in amr2.edges if n2 in e] for n2 in new_n}
                        new_n = [n2 for n2 in new_n if neighborhood==set(neighborhood2[n2])]
                    if len(new_n)==1:
                        bad_node_map[amr1.id][new_n[0]] = n
                    else:
                        raise Exception('Bad node match', amr1.id, n)
                else:
                    bad_node_map[amr1.id][n]=n
    for amr1 in amrs_new:
        amr2 = next(a for a in amrs_old if a.id == amr1.id)
        for n2 in amr2.nodes:
            n = bad_node_map[amr1.id][n2] if amr1.id in bad_node_map else n2
            if amr1.nodes[n] != amr2.nodes[n2]:
                raise Exception('Bad node match', amr1.id, n)
        for e in amr2.edges:
            s2,r,t2 = e
            s = bad_node_map[amr1.id][s2] if amr1.id in bad_node_map else s2
            t = bad_node_map[amr1.id][t2] if amr1.id in bad_node_map else t2
            if (s,r,t) not in amr1.edges:
                raise Exception('Bad edge match', amr1.id, e, amr1.nodes[s],r.amr1.nodes[t])

    print('Node id fixes:',len(bad_node_map), ' '.join(bad_node_map.keys()))
    # subgraphs
    align_file = amr_file_old.replace('.txt','')+'.subgraph_alignments.gold.json'
    sub_aligns = reader.load_alignments_from_json(align_file)
    for amr in amrs_new:
        for align in sub_aligns[amr.id]:
            if amr.id in bad_node_map:
                align.nodes = [bad_node_map[amr.id][n] for n in align.nodes]
        for align in sub_aligns[amr.id]:
            if len(align.nodes)>1:
                for s,r,t in amr.edges:
                    if s in align.nodes and t in align.nodes:
                        align.edges.append((s,r,t))
        sub_aligns[amr.id] = [a for a in sub_aligns[amr.id] if a.nodes]
    align_file = output_file.replace('.txt', '') + '.subgraph_alignments.gold.json'
    print('Writing subgraph alignments to:', align_file)
    write_to_json(align_file, sub_aligns, anonymize=True, amrs=amrs_new)

    # relations
    align_file = amr_file_old.replace('.txt', '') + '.relation_alignments.gold.json'
    rel_aligns = reader.load_alignments_from_json(align_file)
    for amr in amrs_new:
        for align in rel_aligns[amr.id]:
            if amr.id in bad_node_map:
                align.edges = [(bad_node_map[amr.id][s],r,bad_node_map[amr.id][t]) for s,r,t in align.edges]
        for align in rel_aligns[amr.id]:
            sub_align = amr.get_alignment(sub_aligns, token_id=align.tokens[0])
            if sub_align.nodes:
                align.edges = [e for e in align.edges if not (e[0] in sub_align.nodes and e[-1] in sub_align.nodes)]
        rel_aligns[amr.id] = [a for a in rel_aligns[amr.id] if a.edges]
    align_file = output_file.replace('.txt', '') + '.relation_alignments.gold.json'
    print('Writing relation alignments to:', align_file)
    write_to_json(align_file, rel_aligns, anonymize=True, amrs=amrs_new)

    # reentrancies
    align_file = amr_file_old.replace('.txt', '') + '.reentrancy_alignments.gold.json'
    reent_aligns = reader.load_alignments_from_json(align_file)
    for amr in amrs_new:
        reent_aligns[amr.id] = [a for a in reent_aligns[amr.id] if a.edges]
        for align in reent_aligns[amr.id]:
            if amr.id in bad_node_map:
                align.edges = [(bad_node_map[amr.id][s],r,bad_node_map[amr.id][t]) for s,r,t in align.edges]
    align_file = output_file.replace('.txt', '') + '.reentrancy_alignments.gold.json'
    print('Writing reentrancy alignments to:', align_file)
    write_to_json(align_file, reent_aligns, anonymize=True, amrs=amrs_new)

    for amr in amrs_new:
        for n in amr.nodes:
            n_aligned = [a for a in sub_aligns[amr.id] if n in a.nodes]
            if len(n_aligned)!=1:
                raise Exception('Bad node alignment',amr.id,n)
        for e in amr.edges:
            e_aligned = [a for a in sub_aligns[amr.id] if e in a.edges]+\
                        [a for a in rel_aligns[amr.id] if e in a.edges]
            if len(e_aligned)!=1:
                raise Exception('Bad edge alignment',amr.id,e)


if __name__=='__main__':
    main()