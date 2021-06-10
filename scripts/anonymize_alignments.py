import sys

from amr_utils.alignments import write_to_json
from amr_utils.amr_readers import AMR_Reader


def main():
    amr_file = sys.argv[1]
    output_file = sys.argv[2]

    reader = AMR_Reader()
    amrs = reader.load(amr_file)

    # subgraphs
    align_file = amr_file.replace('.txt','')+'.subgraph_alignments.json'
    sub_aligns = reader.load_alignments_from_json(align_file)
    for amr in amrs:
        sub_aligns[amr.id] = [a for a in sub_aligns[amr.id] if a.nodes]
    align_file = output_file.replace('.txt', '') + '.subgraph_alignments.json'
    print('Writing subgraph alignments to:', align_file)
    write_to_json(align_file, sub_aligns, anonymize=True, amrs=amrs)

    # relations
    align_file = amr_file.replace('.txt', '') + '.relation_alignments.json'
    rel_aligns = reader.load_alignments_from_json(align_file)
    for amr in amrs:
        rel_aligns[amr.id] = [a for a in rel_aligns[amr.id] if a.edges]
    align_file = output_file.replace('.txt', '') + '.relation_alignments.json'
    print('Writing relation alignments to:', align_file)
    write_to_json(align_file, rel_aligns, anonymize=True, amrs=amrs)

    # reentrancies
    align_file = amr_file.replace('.txt', '') + '.reentrancy_alignments.json'
    reent_aligns = reader.load_alignments_from_json(align_file)
    for amr in amrs:
        reent_aligns[amr.id] = [a for a in reent_aligns[amr.id] if a.edges]
    align_file = output_file.replace('.txt', '') + '.reentrancy_alignments.json'
    print('Writing reentrancy alignments to:', align_file)
    write_to_json(align_file, reent_aligns, anonymize=True, amrs=amrs)

    for amr in amrs:
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