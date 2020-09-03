import re

from amr_utils.alignments import AMR_Alignment


def preprocess(amr, alignments, align):
    # Certain AMR patterns should always be aligned as a single subgraph, such as named entities.
    # This function adds any mising nodes based on several patterns.

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
        elif s in align.nodes and t not in align.nodes:
            if r==':name' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
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

def preprocess_english(amr, alignments, align):

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
    for s,r,t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            # E.g., "The city of Paris."
            if 'of' in [amr.tokens[t].lower() for t in align.tokens] and amr.nodes[s]=='mean-01' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)


time_re = re.compile('^[0-2]?\d:[0-5]\d$')
def align_fuzzy_match(amr, alignments):

    for n in amr.nodes:
        align = amr.get_alignment(alignments, node_id=n)
        if not align:
            # Try to align attributes in quotes by fuzzy match if only one match exists
            if amr.nodes[n].startswith('"') and amr.nodes[n].endswith('"'):
                label = amr.nodes[n].replace('"', '') #.replace("'",'')
                candidate_strings = [label]
                if len(label)<=1: continue
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
                    if any(l in tokens for l in candidate_strings):
                        candidate_spans.append(span)
                if len(candidate_spans) == 1:
                    span = candidate_spans[0]
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(n)

            # Align other concepts to tokens by fuzzy match if only one exists
            elif amr.nodes[n][0].isalpha() and not amr.nodes[n].endswith('-91') and not amr.get_alignment(alignments, node_id=n):
                label = amr.nodes[n]
                unaligned_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                for prefix_size in [6,5,4]:
                    candidate_tokens = [span for span in unaligned_tokens if '-'.join(amr.lemmas[t] for t in span)[:prefix_size]==label[:prefix_size]]
                    if len(candidate_tokens)!=1:
                        continue
                    candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n2][0].isalpha() and not amr.get_alignment(alignments, node_id=n2) and
                                       not amr.nodes[n2].endswith('-91') and amr.nodes[n2][:prefix_size]==label[:prefix_size]]
                    if len(candidate_nodes)!=1:
                        continue
                    span = candidate_tokens[0]
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(n)
                    break

def align_exact_match_english(amr, alignments):

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

            candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n]==amr.nodes[n2] and not amr.get_alignment(alignments, node_id=n2)]
            if len(candidate_tokens) == 1 and len(candidate_nodes) == 1:
                span = candidate_tokens[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)