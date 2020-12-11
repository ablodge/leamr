from amr_utils.alignments import AMR_Alignment


def add_relation_alignment(amr, relation_alignments, edge, span):
    if not span:
        raise Exception('Tried to align to empty span.')
    for align in relation_alignments[amr.id]:
        if align.tokens == span:
            new_align = align
            new_align.edges.append(edge)
            return
    new_align = AMR_Alignment(type='relation', tokens=span, edges=[edge])
    relation_alignments[amr.id].append(new_align)
    relation_alignments[amr.id] = [align for align in sorted(relation_alignments[amr.id], key=lambda x: x.tokens[0])]


def rule_based_align_relations(amr, subgraph_alignments, relation_alignments):
    if amr.id not in relation_alignments:
        relation_alignments[amr.id] = []
    for s, r, t in amr.edges:
        salign = amr.get_alignment(subgraph_alignments, node_id=s)
        if not salign.tokens:
            continue
        elif s in salign.nodes and t in salign.nodes:
            add_relation_alignment(amr, relation_alignments, (s, r, t), salign.tokens)
        elif any(r.startswith(prefix) for prefix in [':ARG', ':op', ':snt']):
            if r.endswith('-of'):
                talign = amr.get_alignment(subgraph_alignments, node_id=t)
                if not talign: continue
                add_relation_alignment(amr, relation_alignments, (s, r, t), talign.tokens)
            else:
                add_relation_alignment(amr, relation_alignments, (s, r, t), salign.tokens)
        elif r == ':domain':
            add_relation_alignment(amr, relation_alignments, (s, r, t), salign.tokens)
        elif r == ':mod':
            talign = amr.get_alignment(subgraph_alignments, node_id=t)
            if not talign: continue
            add_relation_alignment(amr, relation_alignments, (s, r, t), talign.tokens)


def exact_match_relations(amr, subgraph_alignments, relation_alignments):
    for s, r, t in amr.edges:
        if r.startswith(':prep-'):
            token_label = r.replace(':prep-', '').split('-')
            token_label = ' '.join(token_label)
            candidate_spans = [span for span in amr.spans if
                               ' '.join(amr.lemmas[t].lower() for t in span) == token_label]
            candidate_spans = [span for span in candidate_spans if
                               not amr.get_alignment(subgraph_alignments, token_id=span[0])]
            if len(candidate_spans) == 1:
                add_relation_alignment(amr, relation_alignments, (s, r, t), candidate_spans[0])
        elif r.startswith(':conj-'):
            token_label = r.replace(':conj-', '').split('-')
            token_label = ' '.join(token_label)
            candidate_spans = [span for span in amr.spans if
                               ' '.join(amr.lemmas[t].lower() for t in span) == token_label]
            candidate_spans = [span for span in candidate_spans if
                               not amr.get_alignment(subgraph_alignments, token_id=span[0])]
            if len(candidate_spans) == 1:
                add_relation_alignment(amr, relation_alignments, (s, r, t), candidate_spans[0])
        elif r in [':poss', ':part']:
            token_labels = ["'s", 'of']
            candidate_spans = [span for span in amr.spans if
                               ' '.join(amr.lemmas[t].lower() for t in span) in token_labels]
            candidate_spans = [span for span in candidate_spans if
                               not amr.get_alignment(subgraph_alignments, token_id=span[0])]
            if len(candidate_spans) == 1:
                add_relation_alignment(amr, relation_alignments, (s, r, t), candidate_spans[0])


def rule_based_anchor_relation(edge):
    s, r, t = edge

    if any(r.startswith(prefix) for prefix in [':ARG', ':op', ':snt']) and not r.endswith('-of'):
        return [s]
    elif r in [':domain', ':poss', ':part']:
        return [s]
    # Note: exceptions to -of inversion include
    # :consist-of, :prep-out-of, :prep-on-behalf-of
    elif r.endswith('-of') and not any(
            r.startswith(prefix) for prefix in [':ARG', ':op', ':snt', ':domain', ':poss', ':part']) and \
            r not in [':consist-of', ':prep-out-of', ':prep-on-behalf-of']:
        return [s]
    else:
        return [t]


def normalize_relation(edge):
    s, r, t = edge

    if r.endswith('-of') and r not in [':consist-of', ':prep-out-of', ':prep-on-behalf-of']:
        return t, r.replace('-of', ''), s
    else:
        return edge


def rule_based_align_all_relations(amr, subgraph_alignments):
    for s, r, t in amr.edges:
        anchor = rule_based_anchor_relation((s,r,t))[0]
        align = amr.get_alignment(subgraph_alignments, node_id=anchor)
        if align:
            align.edges.append((s,r,t))