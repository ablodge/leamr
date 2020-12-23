import re

from amr_utils.alignments import AMR_Alignment
from amr_utils.graph_utils import is_rooted_dag, get_rooted_components


def postprocess_subgraph(amr, alignments, align, english=False):
    # Certain AMR patterns should always be aligned as a single subgraph, such as named entities.
    # This function adds any mising nodes based on several patterns.

    if not align.nodes:
        return

    # First pass
    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if amr.nodes[s] == 'name' and r.startswith(':op') and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'date-entity' and r != ':mod' and not r.endswith('-of') and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s].endswith('quantity') and r in [':quant', ':unit'] and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'have-degree-91' and r == ':ARG3' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'have-rel-role-91' and r == ':ARG2' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'have-org-role-91' and r == ':ARG2' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        elif s in align.nodes and t not in align.nodes:
            if amr.nodes[t] == 'have-degree-91' and r == ':ARG3-of' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
            elif amr.nodes[t] == 'have-rel-role-91' and r == ':ARG2-of' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
            elif amr.nodes[t] == 'have-org-role-91' and r == ':ARG2-of' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
    # Second pass
    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if r == ':name' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        if s in align.nodes and t not in align.nodes:
            if amr.nodes[s] == 'name' and r.startswith(':op'):
                talign = amr.get_alignment(alignments, node_id=t)
                if talign:
                    talign.nodes.remove(t)
                align.nodes.append(t)
            elif amr.nodes[s] == 'date-entity' and r != ':mod' and not r.endswith('-of') and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
    if english:
        _postprocess_subgraph_english(amr, alignments, align)


def _postprocess_subgraph_english(amr, alignments, align):
    # postprocessing rules specific to English
    if not align.nodes:
        return

    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if amr.nodes[s] == 'person' and r == ':ARG0-of' and not amr.get_alignment(alignments, node_id=s):
                if amr.nodes[t] in ['have-org-role-91','have-rel-role-91']:
                    align.nodes.append(s)
                elif any(amr.lemmas[align.tokens[-1]].endswith(s) for s in ['er','or','ist']):
                    align.nodes.append(s)
            elif amr.nodes[s] == 'thing' and r in [':ARG0-of',':ARG1-of',':ARG2-of'] and not amr.get_alignment(alignments, node_id=s):
                if any(amr.lemmas[align.tokens[-1]].endswith(s) for s in ['ment', 'tion','sion']):
                    align.nodes.append(s)
                elif r==':ARG1-of':
                    align.nodes.append(s)
            # E.g., "flammable"
            elif amr.nodes[s] == 'possible-01' and not amr.get_alignment(alignments, node_id=s):
                if any(amr.tokens[tok].lower().endswith('able') or amr.tokens[tok].lower().endswith('ible') for tok in
                       align.tokens):
                    align.nodes.append(s)
            # E.g., "highest"
            elif amr.nodes[s] == 'have-degree-91' and not amr.get_alignment(alignments, node_id=s):
                tok = amr.tokens[align.tokens[-1]]
                if r==':ARG2' and (tok.endswith('est') or tok.endswith('er')):
                    align.nodes.append(s)
            # E.g., "Many have criticized the article."
            elif amr.nodes[s] == 'person' and r == ':quant' and not amr.get_alignment(alignments, node_id=s):
                next_tok = align.tokens[-1] + 1
                if next_tok < len(amr.tokens) and amr.lemmas[next_tok].lower() in ['have', 'be']:
                    align.nodes.append(s)
            # E.g., "Many were stolen by burglers."
            elif amr.nodes[s] == 'thing' and r == ':quant' and not amr.get_alignment(alignments, node_id=s):
                next_tok = align.tokens[-1] + 1
                if next_tok < len(amr.tokens) and amr.lemmas[next_tok].lower() in ['have', 'be']:
                    align.nodes.append(s)
        elif s in align.nodes and t not in align.nodes:
            # E.g., "flammable"
            if amr.nodes[t] == 'possible-01' and r == ':ARG1-of' and not amr.get_alignment(alignments, node_id=s):
                if any(amr.tokens[tok].lower().endswith('able') or amr.tokens[tok].lower().endswith('ible') for tok in
                       align.tokens):
                    align.nodes.append(t)
            # imperative
            elif amr.nodes[t] == 'imperative' and r == ':mode' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(t)
            elif amr.nodes[t] == 'you' and r == ':ARG0' and not any(len(span)==1 and amr.lemmas[span[0]] in ['you','your','yours'] for span in amr.spans):
                align.nodes.append(t)
    # third pass
    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            # E.g., "The city of Paris."
            if 'of' in [amr.tokens[t].lower() for t in align.tokens] and \
                    amr.nodes[s] == 'mean-01' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        elif s in align.nodes and t not in align.nodes:
            # if r == ':polarity' and amr.nodes[t] == '-' and not amr.get_alignment(alignments, node_id=t):
            #     token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
            #     if any(token_label.startswith(prefix) for prefix in ['un', 'non']) \
            #             and not any(
            #         token_label.startswith(prefix) for prefix in ['until', 'unite', 'union', 'under', 'nonce']):
            #         prev_tok = align.tokens[0] - 1
            #         if prev_tok < 0: prev_tok = 0
            #         if amr.lemmas[prev_tok] not in ['not', "n't"]:
            #             align.nodes.append(t)
            # E.g., "highest"
            if amr.nodes[s] == 'have-degree-91' and not amr.get_alignment(alignments, node_id=t):
                tok = amr.tokens[align.tokens[-1]]
                if r==':ARG3' and amr.nodes[t] in ['more','most'] and (tok.endswith('est') or tok.endswith('er')):
                    align.nodes.append(t)


def clean_subgraph(amr, alignments, align, english=False):
    # if english:
    #     if len(align.nodes) == 1 and amr.nodes[align.nodes[0]] == 'person' \
    #             and ' '.join(amr.lemmas[t].lower() for t in align.tokens) not in ['person', 'people']:
    #         return None

    if align.nodes and not is_subgraph(amr, align.nodes):
        components = separate_components(amr, align)
        found = False
        for n in amr.nodes:
            if amr.get_alignment(alignments, node_id=n):
                continue
            if all(any(s == n and t in sub.nodes for s, r, t in amr.edges) for sub in components):
                align.nodes.append(n)
                found = True
                break
        if not found:
            return None
    return align


def fuzzy_align_subgraphs(amr, alignments, english=False):
    aligned_nodes = set()

    for n in amr.nodes:
        # exact match names
        if amr.nodes[n] == 'name':
            parts = [(int(r[3:]), t) for s, r, t in amr.edges if s == n and r.startswith(':op')]
            label = ' '.join(amr.nodes[t].replace('"', '') for i, t in sorted(parts, key=lambda x: x[0]))
            candidate_spans = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
            candidate_spans = [span for span in candidate_spans if ' '.join(amr.tokens[t] for t in span)==label]
            if candidate_spans:
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
                aligned_nodes.add(n)
                for _,t in parts:
                    align.nodes.append(t)
                    aligned_nodes.add(t)
    for n in amr.nodes:
        if n in aligned_nodes: continue
        # Try to align attributes in quotes by fuzzy match if only one match exists
        if amr.nodes[n].startswith('"') and amr.nodes[n].endswith('"') or amr.nodes[n][0].isdigit():
            candidate_spans = []
            label = amr.nodes[n].replace('"', '')
            if label.replace('.','') in ['Mr','Mrs','Ms','Dr']:
                continue
            candidate_strings = [label]
            for span in amr.spans:
                tokens = [amr.tokens[t].replace("'", '') for t in span]
                for tok in tokens[:]:
                    if tok and tok[0].isdigit():
                        tokens.append(tok.replace(',', ''))
                if any(l in tokens for l in candidate_strings):
                    candidate_spans.append(span)
            if len(candidate_spans) == 1:
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                if any(amr.nodes[n2] == amr.nodes[n] for n2 in align.nodes):
                    continue
                align.nodes.append(n)
                aligned_nodes.add(n)
    for n in amr.nodes:
        if n in aligned_nodes: continue
        # Align other concepts to tokens by fuzzy match if only one exists
        if amr.nodes[n].replace('"','')[0].isalpha() and not amr.nodes[n].endswith('-91'):
            label = amr.nodes[n].replace('"','')
            if '"' == amr.nodes[n][0] and any(amr.nodes[s]=='name' and t==n and r.startswith(':op') for s,r,t in amr.edges):
                name_node = [s for s,r,t in amr.edges if amr.nodes[s]=='name' and t==n][0]
                name_parts = [t for s,r,t in amr.edges if s==name_node and r.startswith(':op')]
                if len(name_parts)>1:
                    continue
            unaligned_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
            found = False
            # fuzzy prefix match
            for prefix_size in [6, 5, 4]:
                candidate_tokens = [span for span in unaligned_tokens
                                    if '-'.join(amr.lemmas[t] for t in span)[:prefix_size].lower() == label[:prefix_size].lower()
                                    or '-'.join(amr.lemmas[t] for t in span)[:prefix_size].lower() == label.split('-')[0][:prefix_size].lower()]
                if len(candidate_tokens) != 1:
                    continue
                candidate_nodes = [n2 for n2 in amr.nodes if
                                   amr.nodes[n2].replace('"','')[0].isalpha() and not amr.get_alignment(alignments, node_id=n2) and
                                    not amr.nodes[n2].endswith('-91')]
                candidate_nodes = [n2 for n2 in candidate_nodes if
                                   amr.nodes[n2].replace('"','')[:prefix_size] == label[:prefix_size]
                                   or amr.nodes[n2].split('-')[0][:prefix_size] == label.split('-')[0][
                                                                                       :prefix_size]]
                if len(candidate_nodes) != 1:
                    continue
                found = True
                span = candidate_tokens[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
                break
            if found: continue
            # exact match
            candidate_tokens = [span for span in amr.spans if
                                any(amr.lemmas[t] == amr.nodes[n].replace('"','') for t in span)]
            candidate_tokens = [span for span in candidate_tokens
                                if not any(amr.nodes[n2] == amr.nodes[n] for n2 in amr.get_alignment(alignments, token_id=span[0]).nodes)]
            candidate_nodes = [n2 for n2 in amr.nodes if
                               amr.nodes[n] == amr.nodes[n2] and not amr.get_alignment(alignments, node_id=n2)]
            if len(candidate_tokens) == 1 and len(candidate_nodes) == 1:
                span = candidate_tokens[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
    if english:
        _exact_align_subgraphs_english(amr, alignments)


time_re = re.compile('^[0-2]?\d:[0-5]\d$')


def _exact_align_subgraphs_english(amr, alignments):
    # Months, dates, numbers, times
    for n in amr.nodes:
        if amr.nodes[n][0].isdigit():
            align = amr.get_alignment(alignments, node_id=n)
            candidate_spans = []
            if not align:
                label = amr.nodes[n]
                candidate_strings = [label]

                # E.g., "6:00" aligns to 6
                if time_re.match(label):
                    label = label.split(':')[0]
                    oclock = int(label)
                    candidate_strings.append(str(oclock))
                    if oclock > 12:
                        oclock -= 12
                        candidate_strings.append(str(oclock))
                # Months
                if any(t == n and r == ':month' for s, r, t in amr.edges):
                    months = {1: ['January', 'Jan.', 'Jan'],
                              2: ['February', 'Feb.', 'Feb'],
                              3: ['March', 'Mar.', 'Mar'],
                              4: ['April', 'Apr.', 'Apr'],
                              5: ['May'],
                              6: ['June'],
                              7: ['July'],
                              8: ['August', 'Aug.', 'Aug'],
                              9: ['September', 'Sep.', 'Sep'],
                              10: ['October', 'Oct.', 'Oct'],
                              11: ['November', 'Nov.', 'Nov'],
                              12: ['December', 'Dec.', 'Dec']}
                    if label.isdigit() and int(label) in months:
                        candidate_strings.extend(months[int(label)])
                # Numbers
                if amr.nodes[n].isdigit():
                    nums =      {1: 'one',
                                  2: 'two',
                                  3: 'three',
                                  4: 'four',
                                  5: 'five',
                                  6: 'six',
                                  7: 'seven',
                                  8: 'eight',
                                  9: 'nine',
                                  10: 'ten',
                                  11: 'eleven',
                                  12: 'twelve',
                                  13: 'thirteen',
                                  14: 'fourteen',
                                  15: 'fifteen',
                                  16: 'sixteen',
                                  17: 'seventeen',
                                  18: 'eighteen',
                                  19: 'nineteen',
                                  20: 'twenty',
                                  30: 'thirty'}
                    if int(label) in nums:
                        candidate_strings.append(nums[int(label)])

                for span in amr.spans:
                    tokens = [amr.tokens[t].replace("'", '') for t in span]
                    for tok in tokens[:]:
                        if tok and tok[0].isdigit():
                            tokens.append(tok.replace(',', ''))
                    if any(l in tokens for l in candidate_strings):
                        candidate_spans.append(span)
                # Large Numbers
                if label.isdigit() and len(label) >= 3:
                    for span in amr.spans:
                        big_numbers = {'hundred': 100, 'thousand': 1000, 'million': 1e6, 'billion': 1e9,
                                       'trillion': 1e12, 'mill': 1e6, 'bill': 1e9, 'm': 1e6, 'b': 1e9}
                        tokens = [amr.tokens[t].replace("'", '') for t in span]
                        number_tokens = [tok for tok in tokens if tok.isdigit()
                                         or tok.replace('.', '').replace(',', '').isdigit()
                                         or tok.replace('.', '').lower() in big_numbers]
                        if 1 <= len(number_tokens) <= 2:
                            total = 1
                            for num in number_tokens:
                                if num.isdigit():
                                    total *= int(num)
                                elif num.replace('.', '').replace(',', '').isdigit():
                                    try:
                                        total *= float(num.replace(',', ''))
                                    except:
                                        continue
                                else:
                                    num = num.replace('.', '').lower()
                                    total *= big_numbers[num]
                            if str(int(total)) == label:
                                candidate_spans.append(span)
            if len(candidate_spans) == 1:
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                if any(amr.nodes[n2] == amr.nodes[n] for n2 in align.nodes):
                    continue
                align.nodes.append(n)

    # exact match special rules
    for n in amr.nodes:
        align = amr.get_alignment(alignments, node_id=n)
        if not align:
            candidate_tokens = []
            # exact match for 'and' and 'multi-sentence'
            if amr.nodes[n] in ['and', 'multi-sentence']:
                label = amr.nodes[n]
                candidate_tokens = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                if label == 'and':
                    candidate_tokens = [span for span in candidate_tokens if
                                        ' '.join(amr.lemmas[t] for t in span).lower() in ['and', '&', 'additionally', 'as well', 'in addition']]
                    if not candidate_tokens:
                        candidate_tokens = [span for span in candidate_tokens if
                                            ' '.join(amr.lemmas[t] for t in span).lower() in [',', ';']]
                elif label == 'multi-sentence':
                    candidate_tokens = [span for span in candidate_tokens if
                                        len(span) == 1 and amr.lemmas[span[0]] in ['.', ';']]
                    if not candidate_tokens:
                        candidate_tokens = [span for span in candidate_tokens if
                                            len(span) == 1 and amr.lemmas[span[0]] in ['?', '!']]
                    if candidate_tokens and amr.spans[-1] == candidate_tokens[-1]:
                        candidate_tokens.pop()
            # exact match for 'have-03'
            elif amr.nodes[n] == 'have-03':
                candidate_tokens = [span for span in amr.spans if
                                    len(span) == 1 and amr.lemmas[span[0]].lower() in ['have', 'with', "'s"]]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for 'person'
            elif amr.nodes[n] == 'person':
                candidate_tokens = [span for span in amr.spans if
                                    len(span) == 1 and amr.lemmas[span[0]].lower() in ['person', 'people']]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for 'include-91'
            elif amr.nodes[n] == 'include-91':
                candidate_tokens = [span for span in amr.spans if ' '.join(amr.lemmas[t] for t in span).lower() in ['of', 'out of']]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for 'cause-01'
            elif amr.nodes[n] == 'cause-01':
                candidate_tokens = [span for span in amr.spans if
                                    ' '.join(amr.lemmas[t] for t in span).lower() in
                                    ['thus', 'since', 'because', 'cause', 'such', 'such that', 'so', 'therefore',
                                     'out of', 'due to', 'thanks to', 'reason', 'why', 'how', 'consequently', ',']]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]

            # exact match for polarity -
            elif amr.nodes[n] == '-':
                candidate_tokens = [span for span in amr.spans if
                                    ' '.join(amr.lemmas[t] for t in span).lower() in
                                    ['not', "n't", 'non', 'without', 'no', 'none', 'never', 'neither', 'no one']]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for amr-unknown
            elif amr.nodes[n] == 'amr-unknown':
                if any(r==':polarity' and t==n for s,r,t in amr.edges):
                    candidate_tokens = [span for span in amr.spans if ' '.join(amr.lemmas[t] for t in span).lower() in ['?']]
                    candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]

                else:
                    candidate_tokens = [span for span in amr.spans if ' '.join(amr.lemmas[t] for t in span).lower() in
                                        ['why', 'how', 'when', 'where', 'who', 'which', 'what', 'how many','how long','how much']]
                    candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]

            # exact match for rate-entity-91
            elif amr.nodes[n] == 'rate-entity-91':
                candidate_tokens = [span for span in amr.spans if
                                    ' '.join(amr.lemmas[t] for t in span).lower() in
                                    ['per', 'every', 'monthly', 'weekly', 'weekly', 'annually', 'annual']]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for mean-01
            elif amr.nodes[n] == 'mean-01':
                candidate_tokens = [span for span in amr.spans if len(span) == 1 and amr.lemmas[span[0]].lower() in [':', ',',]]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]

            # United States
            elif amr.nodes[n] == 'name' and {amr.nodes[t].replace('"', '') for s, r, t in amr.edges if
                                             s == n and r.startswith(':op')} in [
                {'United', 'States'}, {'America'}, {'United', 'States', 'of', 'America'}]:
                candidate_tokens = [span for span in amr.spans
                                    if not any(amr.nodes[n2].replace('"', '') in ['United', 'States', 'America']
                                               for n2 in amr.get_alignment(alignments, token_id=span[0]).nodes)]
                candidate_tokens = [span for span in candidate_tokens if any(amr.lemmas[t].replace('.', '') in
                                                                             ['American', 'US', 'USA'] for t in span)]
            candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n] == amr.nodes[n2] and not amr.get_alignment(alignments, node_id=n2)]
            if amr.nodes[n] == 'name':
                name = {amr.nodes[t].replace('"', '') for s, r, t in amr.edges if s == n and r.startswith(':op')}
                candidate_nodes = [n2 for n2 in candidate_nodes if
                                   {amr.nodes[t].replace('"', '') for s, r, t in amr.edges if
                                    s == n2 and r.startswith(':op')} == name]
            if len(candidate_tokens) == 1 and len(candidate_nodes) == 1:
                span = candidate_tokens[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
    for span in amr.spans:
        align = amr.get_alignment(alignments, token_id=span[0])
        if not align:
            candidate_tokens = []
            candidate_nodes = []
            label = ' '.join(amr.lemmas[t] for t in span)
            # exact match for 'how'
            if label == 'how':
                candidate_tokens = [s for s in amr.spans if len(s) == 1 and amr.lemmas[s[0]]==label]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]
                candidate_nodes = []
                for n in amr.nodes:
                    if amr.nodes[n]=='thing' and not amr.get_alignment(alignments, node_id=n):
                        if any((s==n and r==':manner-of') or (t=='n' and r==':manner') for s,r,t in amr.edges):
                            candidate_nodes.append(n)
                    elif amr.nodes[n]=='so' and not amr.get_alignment(alignments, node_id=n):
                        so_tokens = [s for s in amr.spans if len(s) == 1 and amr.lemmas[s[0]]=='so']
                        so_tokens = [s for s in so_tokens if not amr.get_alignment(alignments, token_id=s[0])]
                        if not so_tokens:
                            candidate_nodes.append(n)
                    elif amr.nodes[n] == 'have-manner-91' and not amr.get_alignment(alignments, node_id=n):
                        candidate_nodes.append(n)
            # as ... as construction
            elif label == 'as':
                candidate_tokens = [s for s in amr.spans if len(s) == 1 and amr.lemmas[s[0]]==label]
                candidate_tokens = [s for s in candidate_tokens if not amr.get_alignment(alignments, token_id=s[0])]
                candidate_nodes = [n for n in amr.nodes if not amr.get_alignment(alignments, node_id=n)]
                candidate_nodes = [n for n in candidate_nodes if amr.nodes[n] in ['equal']]
                if len(candidate_tokens)<=2 and len(candidate_nodes) == 1 and span==candidate_tokens[0]:
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(candidate_nodes[0])
                    continue
            # try un- non-
            elif len(span)==1 and any(label.startswith(neg) for neg in ['un','non','in']):
                prefix = 'un' if label.startswith('un') else 'in' if label.startswith('in') else 'non'
                candidate_tokens = [span for span in amr.spans if len(span) == 1 and amr.lemmas[span[0]][:6] == label[:6]]
                candidate_tokens = [span for span in candidate_tokens if not amr.get_alignment(alignments, token_id=span[0])]
                candidate_nodes = []
                minus = None
                label = label[len(prefix):]
                for n in amr.nodes:
                    if amr.nodes[n].split('-')[0][:4]==label[:4]:
                        m = [t for s,r,t in amr.edges if s==n and r==':polarity' and amr.nodes[t]=='-']
                        if m:
                            candidate_nodes.append(n)
                            minus = m[0]
                if len(candidate_tokens) == 1 and len(candidate_nodes) == 1:
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(candidate_nodes[0])
                    align.nodes.append(minus)
            if len(candidate_tokens) == 1 and len(candidate_nodes) == 1:
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(candidate_nodes[0])



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
        if amr.nodes[nodes[0]] == amr.nodes[nodes[1]]:
            return False
        if parents1 == parents2 and len(parents1) == 1 and not children:
            return True
    return False
