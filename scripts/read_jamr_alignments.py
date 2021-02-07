from amr_utils.alignments import write_to_json, AMR_Alignment
from amr_utils.amr_readers import LDC_AMR_Reader, JAMR_AMR_Reader


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
    amr_file2 = 'data/szubert/szubert_amrs.txt'

    reader = LDC_AMR_Reader('jamr')
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

    reader = JAMR_AMR_Reader()
    amrs2 = reader.load(amr_file2, remove_wiki=True)

    amrs = {amr.id:amr for amr in amrs}
    amrs2 = {amr.id: amr for amr in amrs2}

    for amr_id in amrs:
        translate_labels = {}
        amr = amrs[amr_id]
        amr2 = amrs2[amr_id]
        for n in amr.nodes:
            if n not in amr2.nodes or amr.nodes[n]!=amr2.nodes[n]:
                candidates = [n2 for n2 in amr2.nodes if amr.nodes[n]==amr2.nodes[n2]]
                candidates = [n2 for n2 in candidates if n2 not in amr.nodes or amr.nodes[n2]!=amr2.nodes[n2]]
                neighbors = {n2:[f'{amr2.nodes[s]}_{r}_{amr2.nodes[t]}' for s,r,t in amr2.edges if n2 in [s,t]] for n2 in candidates}
                n_neighbors = [f'{amr.nodes[s]}_{r}_{amr.nodes[t]}' for s,r,t in amr.edges if n in [s,t]]
                candidates = [n2 for n2 in candidates if set(neighbors[n2])==set(n_neighbors)]

                neighbors2 = {n2:[s for s,r,t in amr2.edges if n2==t]+[t for s,r,t in amr2.edges if n2==s] for n2 in candidates}
                neighbors2 = {n2: [f'{amr2.nodes[s]}_{r}_{amr2.nodes[t]}'
                                   for s, r, t in amr2.edges if s in neighbors2[n2] or t in neighbors2[n2]] for n2 in candidates}
                n_neighbors2 = [s for s,r,t in amr.edges if n==t]+[t for s,r,t in amr.edges if n==s]
                n_neighbors2 = [f'{amr.nodes[s]}_{r}_{amr.nodes[t]}'
                                   for s, r, t in amr.edges if s in n_neighbors2 or t in n_neighbors2]
                candidates = [n2 for n2 in candidates if set(neighbors2[n2]) == set(n_neighbors2)]
                if len(candidates)==1:
                    translate_labels[n] = candidates[0]
                else:
                    raise Exception()
        for align in alignments[amr_id]:
            for i,n in enumerate(align.nodes):
                if n in translate_labels:
                    align.nodes[i] = translate_labels[n]
            for n in align.nodes:
                if n not in amr2.nodes:
                    raise Exception()
            align.amr = amr2
        if amr.tokens!=amr2.tokens:
            tokens = amr.tokens.copy()
            tokens2=amr2.tokens.copy()
            for i,t in enumerate(tokens2):
                if t[0]=='@' and t[-1]=='@' and len(t)==3:
                    tokens2[i] = t[1]
                elif t=='``':
                    tokens2[i] = '"'
            offset = 0
            translate_tokens = {}
            partial = ''
            for i,t in enumerate(tokens):
                j = i+offset
                if tokens[i]==tokens2[j]:
                    translate_tokens[i]=j
                elif len(tokens2[j])>len(tokens[i]) and tokens[i] in tokens2[j]:
                    translate_tokens[i] = j
                    partial+=tokens[i]
                    if len(partial)<len(tokens2[j]):
                        offset-=1
                    else:
                        partial = ''
            for align in alignments[amr_id]:
                new_tokens = [translate_tokens[t] for t in align.tokens]
                new_tokens = set(new_tokens)
                new_tokens = [t for t in sorted(new_tokens)]
                if align.tokens!=new_tokens:
                    align.tokens = new_tokens


    write_to_json('data/szubert/szubert_amrs.jamr_alignments.json', alignments)


if __name__=='__main__':
    main()