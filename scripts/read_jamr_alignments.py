from amr_utils.alignments import AMR_Alignment
from amr_utils.amr_readers import AMR_Reader


def clean_alignments(amr, alignments, spans):
    aligns = []
    for span in spans:
        align = amr.get_alignment(alignments, token_id=span[0])
        if align:
            aligns.append(align)
        else:
            aligns.append(AMR_Alignment(type='subgraph', tokens=span, amr=amr))
    alignments[amr.id] = aligns


def main():
    amr_file = 'data/szubert/szubert_amrs.jamr_alignments.txt'
    # amr_file2 = 'data/szubert/szubert_amrs.txt'

    reader = AMR_Reader()
    amrs, alignments = reader.load(amr_file, remove_wiki=True, output_alignments=True)
    for amr in amrs:
        spans = set()
        taken = set()
        for align in alignments[amr.id]:
            align.type = 'subgraph'
            align.amr = amr
            spans.add(tuple(align.tokens))
            taken.update(align.tokens)
        for t in range(len(amr.tokens)):
            if t not in taken:
                spans.add((t,))
        spans = [list(span) for span in sorted(spans, key=lambda x:x[0])]
        clean_alignments(amr, alignments, spans)

    reader.save_alignments_to_json('data/szubert/szubert_amrs.jamr_alignments.json', alignments)

if __name__=='__main__':
    main()