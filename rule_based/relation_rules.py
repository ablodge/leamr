from amr_utils.alignments import AMR_Alignment


def add_relation_alignment(amr, relation_alignments, edge, span):
    for align in relation_alignments[amr.id]:
        if align.span == span:
            new_align = align
            new_align.edges.append(edge)
            return
    new_align = AMR_Alignment(type='relation', tokens=span, edges=[edge])
    relation_alignments[amr.id].append(new_align)
    relation_alignments[amr.id] = [align for align in sorted(relation_alignments[amr.id], key=lambda x:x.tokens[0])]

def rule_based_align_all_relations(amr, subgraph_alignments, relation_alignments):
    if amr.id not in relation_alignments:
        relation_alignments[amr.id] = []
    for s,r,t in amr.edges:
        # Note: exceptions to -of inversion include
        # :consist-of, :prep-out-of, :prep-on-behalf-of
        if any(r.startswith(prefix) for prefix in [':ARG',':op',':snt']) and not r.endswith('-of'):
            salign = amr.get_alignment(subgraph_alignments, node_id=s)
            add_relation_alignment(amr, relation_alignments, (s,r,t), salign.tokens)
        elif r in [':domain',':poss',':part']:
            salign = amr.get_alignment(subgraph_alignments, node_id=s)
            add_relation_alignment(amr, relation_alignments, (s,r,t), salign.tokens)
        elif r.endswith('-of') and not any(r.startswith(prefix) for prefix in [':ARG',':op',':snt',':domain',':poss',':part']) and \
            r not in [':consist-of', ':prep-out-of', ':prep-on-behalf-of']:
            salign = amr.get_alignment(subgraph_alignments, node_id=s)
            add_relation_alignment(amr, relation_alignments, (s, r, t), salign.tokens)
        else:
            talign = amr.get_alignment(subgraph_alignments, node_id=t)
            add_relation_alignment(amr, relation_alignments, (s,r,t), talign.tokens)

def exact_match_align_prepositions(amr, subgraph_alignments, relation_alignments):
    for s, r, t in amr.edges:
        if r.startswith(':prep-'):
            token_label = r.replace(':prep-','').split('-')
            token_label = ' '.join(token_label)
            candidate_spans = [span for span in amr.spans if ' '.join(amr.lemmas[t].lower() for t in span)==token_label]
            candidate_spans = [span for span in candidate_spans if not amr.get_alignment(subgraph_alignments, token_id=span[0])]
            if len(candidate_spans) == 1:
                add_relation_alignment(amr, relation_alignments, (s,r,t), candidate_spans[0])
        elif r.startswith(':conj-'):
            token_label = r.replace(':conj-', '').split('-')
            token_label = ' '.join(token_label)
            candidate_spans = [span for span in amr.spans if ' '.join(amr.lemmas[t].lower() for t in span) == token_label]
            candidate_spans = [span for span in candidate_spans if not amr.get_alignment(subgraph_alignments, token_id=span[0])]
            if len(candidate_spans) == 1:
                add_relation_alignment(amr, relation_alignments, (s, r, t), candidate_spans[0])
