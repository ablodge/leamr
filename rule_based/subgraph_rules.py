import re

from amr_utils.alignments import AMR_Alignment
from amr_utils.graph_utils import is_rooted_dag, get_rooted_components


def postprocess_subgraph(amr, alignments, align, english=False):
    # Certain AMR patterns should always be aligned as a single subgraph, such as named entities.
    # This function adds any mising nodes based on several patterns.

    if not align.nodes:
        return

    # First pass
    for s,r,t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if amr.nodes[s]=='name' and r.startswith(':op') and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s]=='date-entity' and r!=':mod' and not r.endswith('-of') and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s].endswith('quantity') and r in [':quant',':unit'] and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            if amr.nodes[s] == 'person' and r==':ARG0-of' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'thing' and r==':ARG1-of' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'have-degree-91' and r==':ARG3' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'have-rel-role-91' and r==':ARG2' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        # elif s in align.nodes and t not in align.nodes:
        #     if r==':name' and not amr.get_alignment(alignments, node_id=t):
        #         align.nodes.append(t)
    # Second pass
    for s,r,t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if r==':name' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        if s in align.nodes and t not in align.nodes:
            if amr.nodes[s]=='name' and r.startswith(':op'):
                talign = amr.get_alignment(alignments, node_id=t)
                if talign:
                    talign.nodes.remove(t)
                align.nodes.append(t)
            elif amr.nodes[s]=='date-entity' and r!=':mod' and not r.endswith('-of') and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
    # Never align multi-sentence to the last token
    # if len(align.nodes) == 1:
    #     n = align.nodes[0]
    #     if amr.nodes[n] == 'multi-sentence':
    #         if align.tokens[0] == len(amr.tokens)-1:
    #             align.nodes.clear()
    if english:
        _postprocess_subgraph_english(amr, alignments, align)

def _postprocess_subgraph_english(amr, alignments, align):
    # postprocessing rules specific to English
    if not align.nodes:
        return

    for s,r,t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            # E.g., "unimaginable"
            if amr.nodes[s]=='possible-01' and not amr.get_alignment(alignments, node_id=s):
                if any(amr.tokens[tok].lower().endswith('able') or amr.tokens[tok].lower().endswith('ible') for tok in align.tokens):
                    align.nodes.append(s)
            # E.g., "Many have criticized the article."
            elif amr.nodes[s] == 'person' and r==':quant' and not amr.get_alignment(alignments, node_id=s):
                next_tok = align.tokens[-1]+1
                if next_tok<len(amr.tokens) and amr.lemmas[next_tok].lower() in ['have','be']:
                    align.nodes.append(s)
            # E.g., "Many were stolen by burglers."
            elif amr.nodes[s] == 'thing' and r==':quant' and not amr.get_alignment(alignments, node_id=s):
                next_tok = align.tokens[-1] + 1
                if next_tok<len(amr.tokens) and amr.lemmas[next_tok].lower() in ['have','be']:
                    align.nodes.append(s)
        if s in align.nodes and t not in align.nodes:
            # E.g., "unimaginable"
            if amr.nodes[t] == 'possible-01' and r==':ARG1-of' and not amr.get_alignment(alignments, node_id=s):
                if any(amr.tokens[tok].lower().endswith('able') or amr.tokens[tok].lower().endswith('ible') for tok in
                       align.tokens):
                    align.nodes.append(t)
    for s,r,t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            # E.g., "The city of Paris."
            if 'of' in [amr.tokens[t].lower() for t in align.tokens] and amr.nodes[s]=='mean-01' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        elif s in align.nodes and t not in align.nodes:
            if r==':polarity' and amr.nodes[t]=='-' and not amr.get_alignment(alignments, node_id=t):
                token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
                if any(token_label.startswith(prefix) for prefix in ['un','non']) \
                        and not any(token_label.startswith(prefix) for prefix in ['until','unite','union','under','nonce']):
                    prev_tok = align.tokens[0] - 1
                    if prev_tok < 0: prev_tok = 0
                    if amr.lemmas[prev_tok] not in ['not',"n't"]:
                        align.nodes.append(t)


def clean_subgraph(amr, alignments, align, english=False):
    if len(align.nodes) == 1 and amr.nodes[align.nodes[0]]=='multi-sentence' and align.tokens == [len(amr.tokens)-1]:
        return None
    if english:
        if len(align.nodes) == 1 and amr.nodes[align.nodes[0]]=='person' \
            and ' '.join(amr.lemmas[t].lower() for t in align.tokens) not in ['person','people']:
            return None

    if align.nodes and not is_subgraph(amr, align.nodes):
        components = separate_components(amr, align)
        found = False
        for n in amr.nodes:
            if amr.get_alignment(alignments, node_id=n):
                continue
            if all(any(s==n and t in sub.nodes for s,r,t in amr.edges) for sub in components):
                align.nodes.append(n)
                found = True
                break
        if not found:
            return None
    return align


time_re = re.compile('^[0-2]?\d:[0-5]\d$')
def fuzzy_align_subgraphs(amr, alignments, english=False):

    for n in amr.nodes:
        # Try to align attributes in quotes by fuzzy match if only one match exists
        if amr.nodes[n].startswith('"') and amr.nodes[n].endswith('"') or amr.nodes[n][0].isdigit():
            align = amr.get_alignment(alignments, node_id=n)
            if not align:
                label = amr.nodes[n].replace('"', '') #.replace("'",'')
                candidate_strings = [label]

                # E.g., "6:00" aligns to 6
                if time_re.match(label):
                    label = label.split(':')[0]
                    oclock = int(label)
                    candidate_strings.append(str(oclock))
                    if oclock>12:
                        oclock-=12
                        candidate_strings.append(str(oclock))
                candidate_spans = []
                for span in amr.spans:
                    tokens = [amr.tokens[t].replace("'",'') for t in span]
                    for tok in tokens[:]:
                        if tok and tok[0].isdigit():
                            tokens.append(tok.replace(',',''))
                    if any(l in tokens for l in candidate_strings):
                        candidate_spans.append(span)
                if len(candidate_spans) == 1 or (candidate_spans and any(amr.nodes[s]=='name' and r.startswith(':op') and t==n for s,r,t in amr.edges)):
                    span = candidate_spans[0]
                    align = amr.get_alignment(alignments, token_id=span[0])
                    if any(amr.nodes[n2]==amr.nodes[n] for n2 in align.nodes):
                        continue
                    align.nodes.append(n)
    for n in amr.nodes:
        # Align other concepts to tokens by fuzzy match if only one exists
        if amr.nodes[n][0].isalpha() and not amr.nodes[n].endswith('-91') and not amr.get_alignment(alignments, node_id=n):
            align = amr.get_alignment(alignments, node_id=n)
            if not align:
                label = amr.nodes[n]
                unaligned_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                found = False
                for prefix_size in [6,5,4]:
                    candidate_tokens = [span for span in unaligned_tokens
                                        if '-'.join(amr.lemmas[t] for t in span)[:prefix_size]==label[:prefix_size]
                                        or '-'.join(amr.lemmas[t] for t in span)[:prefix_size]==label.split('-')[0][:prefix_size]]
                    if len(candidate_tokens)!=1:
                        continue
                    candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n2][0].isalpha() and not amr.get_alignment(alignments, node_id=n2) and
                                       not amr.nodes[n2].endswith('-91')]
                    candidate_nodes = [n2 for n2 in candidate_nodes if amr.nodes[n2][:prefix_size]==label[:prefix_size]
                                       or amr.nodes[n2].split('-')[0][:prefix_size]==label.split('-')[0][:prefix_size]]
                    if len(candidate_nodes)!=1:
                        continue
                    found = True
                    span = candidate_tokens[0]
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(n)
                    break
                if found: continue
                candidate_tokens = [span for span in amr.spans if any(amr.lemmas[t].lower()==amr.nodes[n] for t in span)]
                candidate_tokens = [span for span in candidate_tokens
                                    if not any(amr.nodes[n2]==amr.nodes[n]
                                               for n2 in amr.get_alignment(alignments, token_id=span[0]).nodes)]
                candidate_nodes = [n2 for n2 in amr.nodes if
                                   amr.nodes[n] == amr.nodes[n2] and not amr.get_alignment(alignments, node_id=n2)]
                if len(candidate_tokens) == 1 and len(candidate_nodes) == 1:
                    span = candidate_tokens[0]
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(n)
    if english:
        _exact_align_subgraphs_english(amr, alignments)

def _exact_align_subgraphs_english(amr, alignments):

    for n in amr.nodes:
        align = amr.get_alignment(alignments, node_id=n)
        if not align:
            candidate_tokens = []
            # exact match for 'and' and 'multi-sentence'
            if amr.nodes[n] in ['and', 'multi-sentence']:
                label = amr.nodes[n]
                candidate_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                if label == 'and':
                    candidate_tokens = [span for span in candidate_tokens if len(span)==1 and amr.lemmas[span[0]].lower() in ['and','&',',',';']]
                elif label == 'multi-sentence':
                    candidate_tokens = [span for span in candidate_tokens if len(span) == 1 and amr.lemmas[span[0]] in ['.', '!', ';']]
                    if candidate_tokens and amr.spans[-1] == candidate_tokens[-1]:
                        candidate_tokens.pop()
            # exact match for 'have-03'
            elif amr.nodes[n] == 'have-03':
                candidate_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                candidate_tokens = [span for span in candidate_tokens if len(span)==1 and amr.lemmas[span[0]].lower() in ['have','with',"'s"]]
            # exact match for 'include-91'
            elif amr.nodes[n] == 'include-91':
                candidate_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                candidate_tokens = [span for span in candidate_tokens if len(span)==1 and amr.lemmas[span[0]].lower() in ['of','out of']]
            # exact match for 'cause-01'
            elif amr.nodes[n] == 'cause-01':
                candidate_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                candidate_tokens = [span for span in candidate_tokens if len(span) == 1 and amr.lemmas[span[0]].lower() in
                                    ['thus', 'since', 'because', 'cause', 'such', 'such that', 'so', 'therefore',
                                     'out of', 'due to', 'thanks to', 'reason', 'why', 'consequently']]
            # exact match for polarity -
            elif amr.nodes[n] == '-':
                candidate_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                candidate_tokens = [span for span in candidate_tokens if
                                    len(span) == 1 and amr.lemmas[span[0]].lower() in
                                    ['not', "n't", 'non', 'without', 'no', 'none', 'never', 'neither', 'no one']]
            # exact match for mean-01
            elif amr.nodes[n] == 'mean-01':
                candidate_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                candidate_tokens = [span for span in candidate_tokens if len(span) == 1 and amr.lemmas[span[0]].lower() in [':']]
            # United States
            elif amr.nodes[n] == 'name' and {amr.nodes[t].replace('"','') for s,r,t in amr.edges if s==n and r.startswith(':op')} in [
                {'United', 'States'}, {'America'}, {'United', 'States', 'of', 'America'}]:
                candidate_tokens = [span for span in amr.spans
                                    if not any(amr.nodes[n2].replace('"','') in ['United', 'States', 'America']
                                               for n2 in amr.get_alignment(alignments, token_id=span[0]).nodes)]
                candidate_tokens = [span for span in candidate_tokens if any(amr.lemmas[t].replace('.','') in
                                    ['American', 'US', 'USA'] for t in span)]
            candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n]==amr.nodes[n2] and not amr.get_alignment(alignments, node_id=n2)]
            if amr.nodes[n] == 'name':
                name = {amr.nodes[t].replace('"','') for s,r,t in amr.edges if s==n and r.startswith(':op')}
                candidate_nodes = [n2 for n2 in candidate_nodes if
                                   {amr.nodes[t].replace('"','') for s,r,t in amr.edges if s==n2 and r.startswith(':op')} == name]
            if len(candidate_tokens) == 1 and len(candidate_nodes) == 1:
                span = candidate_tokens[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)

def separate_components(amr, align):
    node_labels = [amr.nodes[n] for n in align.nodes]
    if len(node_labels) > 1 and all(node == node_labels[0] for node in node_labels):
        return [AMR_Alignment(type='subgraph', tokens=align.tokens, nodes=[n], amr=amr) for n in align.nodes]
    if not align.nodes:
        return [align]
    if is_subgraph(amr, align.nodes):
        return [align]
    sub = amr.get_subgraph(align.nodes)
    components = get_rooted_components(sub)
    components = [list(sub.nodes.keys()) for sub in components]
    components = [AMR_Alignment(type='subgraph', tokens=align.tokens, nodes=nodes, amr=amr) for nodes in components]
    return components

def is_subgraph(amr, nodes):
    subamr = amr.get_subgraph(nodes)
    if is_rooted_dag(subamr):
        return True
    # handle "never => ever, -" and other similar cases
    if len(subamr.nodes) == 2:
        nodes = [n for n in subamr.nodes]
        parents1 = [s for s, r, t in amr.edges if t == nodes[0]]
        parents2 = [s for s, r, t in amr.edges if t == nodes[1]]
        children = [t for s, r, t in amr.edges if s in nodes]
        # rels = [r for s, r, t in amr.edges if t in nodes]
        if amr.nodes[nodes[0]]==amr.nodes[nodes[1]]:
            return False
        if parents1 == parents2 and len(parents1) == 1 and not children:
            return True
    return False