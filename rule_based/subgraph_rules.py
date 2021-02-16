import re
from collections import Counter

from amr_utils.alignments import AMR_Alignment
from amr_utils.graph_utils import is_rooted_dag, get_connected_components


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
            elif amr.nodes[s] == 'date-entity' and r != ':mod' and not r.endswith('-of') and not amr.get_alignment(
                    alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s].endswith('-quantity') and r in [':quant', ':unit'] and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s].endswith('-entity') and r == ':value' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] in ['have-degree-91', 'have-quant-91'] and r == ':ARG3' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] in ['have-rel-role-91', 'have-org-role-91'] and r in [':ARG2',':ARG3'] and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            elif amr.nodes[s] == 'relative-position' and r == ':direction' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        elif s in align.nodes and t not in align.nodes:
            if r == ':name' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
            elif amr.nodes[t] in ['have-degree-91', 'have-quant-91'] and r == ':ARG3-of' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
            elif amr.nodes[t] in ['have-rel-role-91', 'have-org-role-91'] and r in [':ARG2-of',':ARG3-of'] and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
    # Second pass
    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if r == ':name' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
            if amr.nodes[s] == 'publication-91' and r == ':ARG1' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        if s in align.nodes and t not in align.nodes:
            if amr.nodes[s] == 'name' and r.startswith(':op'):
                talign = amr.get_alignment(alignments, node_id=t)
                if talign:
                    talign.nodes.remove(t)
                align.nodes.append(t)
            elif amr.nodes[s] == 'date-entity' and r != ':mod' and not r.endswith('-of') and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
            elif amr.nodes[s].endswith('-quantity') and r == ':unit' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
    # Third pass
    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if amr.nodes[s] == 'publication-91' and r == ':ARG1' and amr.nodes[t] in ['publication', 'book',
                                                                                      'newspaper'] and not amr.get_alignment(
                    alignments, node_id=s):
                align.nodes.append(s)
    if english:
        _postprocess_subgraph_english(amr, alignments, align)


def _postprocess_subgraph_english(amr, alignments, align):
    # postprocessing rules specific to English
    if not align.nodes:
        return

    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            if amr.nodes[s] == 'person' and r == ':ARG0-of' and not amr.get_alignment(alignments, node_id=s):
                if amr.nodes[t] in ['have-org-role-91', 'have-rel-role-91']:
                    align.nodes.append(s)
                elif any(amr.lemmas[align.tokens[-1]].endswith(s) for s in ['er', 'or', 'ist']):
                    align.nodes.append(s)
            elif amr.nodes[s] == 'thing' and r in [':ARG0-of', ':ARG1-of', ':ARG2-of'] and not amr.get_alignment(
                    alignments, node_id=s):
                if any(amr.lemmas[align.tokens[-1]].endswith(s) for s in ['ment', 'tion', 'sion']):
                    align.nodes.append(s)
                elif r == ':ARG1-of':
                    align.nodes.append(s)
            # E.g., "flammable"
            elif amr.nodes[s] == 'possible-01' and not amr.get_alignment(alignments, node_id=s):
                if any(amr.lemmas[tok].lower().endswith('able') or amr.lemmas[tok].lower().endswith('ible') for tok in
                       align.tokens):
                    align.nodes.append(s)
            # E.g., "highest"
            elif amr.nodes[s] == 'have-degree-91' and not amr.get_alignment(alignments, node_id=s):
                tok = amr.tokens[align.tokens[-1]]
                if r == ':ARG2' and (tok.endswith('est') or tok.endswith('er')):
                    align.nodes.append(s)
            # E.g., "Many have criticized the article."
            elif amr.nodes[s] == 'person' and r == ':quant' and not amr.get_alignment(alignments, node_id=s):
                next_tok = align.tokens[-1] + 1
                if next_tok < len(amr.lemmas) and amr.lemmas[next_tok].lower() in ['have', 'be']:
                    align.nodes.append(s)
            # E.g., "Many were stolen by burglers."
            elif amr.nodes[s] == 'thing' and r == ':quant' and not amr.get_alignment(alignments, node_id=s):
                next_tok = align.tokens[-1] + 1
                if next_tok < len(amr.lemmas) and amr.lemmas[next_tok].lower() in ['have', 'be']:
                    align.nodes.append(s)
            elif amr.nodes[s] in ['after','before'] and r == ':op1'  and amr.nodes[t]=='now' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        elif s in align.nodes and t not in align.nodes:
            # imperative
            if amr.nodes[t] == 'imperative' and r == ':mode' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
            elif amr.nodes[t] == 'you' and r == ':ARG0' \
                    and not any(
                len(span) == 1 and amr.lemmas[span[0]] in ['you', 'your', 'yours', "y'all"] for span in amr.spans) \
                    and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
            # E.g., "flammable"
            elif amr.nodes[t] == 'possible-01' and r == ':ARG1-of' and not amr.get_alignment(alignments, node_id=t):
                if any(amr.lemmas[tok].lower().endswith('able') or amr.lemmas[tok].lower().endswith('ible') for tok in
                       align.tokens):
                    align.nodes.append(t)
            elif amr.nodes[s] in ['after','before'] and r == ':op1'  and amr.nodes[t]=='now' and not amr.get_alignment(alignments, node_id=t):
                align.nodes.append(t)
    # second pass
    for s, r, t in amr.edges:
        if t in align.nodes and s not in align.nodes:
            # E.g., "The city of Paris."
            if 'of' in [amr.tokens[t].lower() for t in align.tokens] and \
                    amr.nodes[s] == 'mean-01' and not amr.get_alignment(alignments, node_id=s):
                align.nodes.append(s)
        elif s in align.nodes and t not in align.nodes:
            # E.g., "highest"
            if amr.nodes[s] == 'have-degree-91' and not amr.get_alignment(alignments, node_id=t):
                tok = amr.tokens[align.tokens[-1]]
                if r == ':ARG3' and amr.nodes[t] in ['more', 'most'] and (tok.endswith('est') or tok.endswith('er')):
                    align.nodes.append(t)


def normalize_token_label(amr, tokens):
    tokens = [amr.tokens[t] for t in tokens]
    token_label = ' '.join([tok for tok in tokens if tok!='"'])
    if token_label.startswith('<a_href'):
        token_label = token_label.replace('<a_href="', '').replace('">', '')
    if token_label == 'British':
        token_label = 'Britain'
    elif token_label == 'French':
        return 'France'
    elif token_label == 'Italian':
        return 'Italy'
    elif token_label == 'Chinese':
        return 'China'
    elif token_label == 'Japanese':
        return 'Japan'
    elif token_label == 'Canadian':
        return 'Canada'
    elif token_label == 'German':
        return 'Germany'
    elif token_label in ['Korean', 'Russian', 'Australian', 'Austrian']:
        return token_label[:-1]
    elif token_label in ['Israeli', 'Iraqi', 'Irani',]:
        return token_label[:-1]
    return token_label


def normalize_lemma_label(amr, tokens):
    token_label = '-'.join(amr.lemmas[t] for t in tokens)
    if token_label.startswith('<a_href'):
        token_label = token_label.replace('<a_href="', '').replace('">', '')
    if token_label == 'daily':
        return 'day'
    return token_label


def fuzzy_align_subgraphs(amr, alignments, english=False):

    aligned_nodes = set()

    # exact match names
    for n in amr.nodes:
        if amr.nodes[n] == 'name':
            parts = [(int(r[3:]), t) for s, r, t in amr.edges if s == n and r.startswith(':op')]
            parts = [t for r, t in sorted(parts, key=lambda x: x[0])]
            label = ' '.join(amr.nodes[t].replace('"', '') for t in parts)
            candidate_spans = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
            candidate_spans = [span for span in candidate_spans if normalize_token_label(amr, span).lower() == label.lower()]
            if candidate_spans:
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
                aligned_nodes.add(n)
                for t in parts:
                    align.nodes.append(t)
                    aligned_nodes.add(t)
            elif len(parts) > 1:
                # look for acronym
                acronym = ''
                for t in parts:
                    letter = amr.nodes[t].replace('"', '')[0]
                    if letter.isalpha() and letter.isupper():
                        acronym += letter
                if len(acronym) >= 2:
                    candidate_spans = [span for span in amr.spans if
                                       not amr.get_alignment(alignments, token_id=span[0])]
                    candidate_spans = [span for span in candidate_spans if
                                       len(span) == 1 and normalize_token_label(amr, span) == acronym]
                    if candidate_spans:
                        span = candidate_spans[0]
                        align = amr.get_alignment(alignments, token_id=span[0])
                        for t in parts:
                            align.nodes.append(t)
                            aligned_nodes.add(t)
                        continue
                # look for incorrect spans
                for start in range(len(amr.tokens)):
                    span = [t for t in range(start, start + len(parts))]
                    if span[-1] >= len(amr.tokens): continue
                    if amr.get_alignment(alignments, token_id=span[0]): continue
                    if normalize_token_label(amr, span) == label:
                        tok = span[0]
                        if any(len(amr.tokens[t]) >= 4 for t in span):
                            while len(amr.tokens[tok]) <= 3:
                                tok += 1
                        align = amr.get_alignment(alignments, token_id=tok)
                        if align: continue
                        for t in parts:
                            align.nodes.append(t)
                            aligned_nodes.add(t)
                        break
    # Single token match for attributes
    for n in amr.nodes:
        if n in aligned_nodes: continue
        if amr.nodes[n].startswith('"') and amr.nodes[n].endswith('"') or amr.nodes[n][0].isdigit():
            candidate_spans = []
            label = node_label(amr, n)
            if any(amr.nodes[s] == 'name' and t == n and r.startswith(':op') for s, r, t in amr.edges):
                continue
            for span in amr.spans:
                tokens = [amr.tokens[t].replace("'", '') for t in span]
                for tok in tokens[:]:
                    if tok and tok[0].isdigit():
                        tokens.append(tok.replace(',', ''))
                if label in tokens:
                    candidate_spans.append(span)
            if len(candidate_spans) == 1:
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                if any(amr.nodes[n2] == amr.nodes[n] for n2 in align.nodes):
                    continue
                align.nodes.append(n)
                aligned_nodes.add(n)

    is_name = lambda n: '"' == amr.nodes[n][0] and any(amr.nodes[s] == 'name' and t == n and r.startswith(':op') for s, r, t in amr.edges)

    # Fuzzy prefix match for 6, 5, or 4 characters
    for n in amr.nodes:
        prefix_sizes = [6, 5, 4]
        if n in aligned_nodes: continue
        if amr.nodes[n].replace('"', '')[0].isalpha() and not amr.nodes[n].endswith('-91'):
            label = node_label(amr, n)
            if is_name(n):
                name_node = [s for s, r, t in amr.edges if amr.nodes[s] == 'name' and t == n][0]
                name_parts = [(r, t) for s, r, t in amr.edges if s == name_node and r.startswith(':op')]
                name_parts = [t for r,t in sorted(name_parts, key=lambda x:int(x[0][3:]))]
                if len(name_parts) > 1:
                    label = '-'.join(amr.nodes[p] for p in name_parts).replace('"','')
                    prefix_sizes = [10,9,8]
            # fuzzy prefix match
            for prefix_size in prefix_sizes:
                candidate_spans = [span for span in amr.spans
                                    if normalize_lemma_label(amr, span)[:prefix_size].lower() == label[:prefix_size].lower()]
                if not is_name(n):
                    candidate_spans += [span for span in amr.spans
                                    if normalize_lemma_label(amr, span)[:prefix_size].lower() == label.split('-')[0][:prefix_size].lower()
                                    and span not in candidate_spans]
                candidate_spans = [span for span in candidate_spans if not amr.get_alignment(alignments, token_id=span[0])]
                if len(candidate_spans) != 1:
                    continue
                candidate_nodes = [n2 for n2 in amr.nodes if
                                   amr.nodes[n2].replace('"', '')[0].isalpha() and not amr.get_alignment(alignments, node_id=n2) and
                                   not amr.nodes[n2].endswith('-91')]
                candidate_nodes = [n2 for n2 in candidate_nodes if
                                   node_label(amr, n2)[:prefix_size] == label[:prefix_size]
                                   or amr.nodes[n2].split('-')[0][:prefix_size] == label.split('-')[0][:prefix_size]]
                if is_name(n):
                    candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n2]=='name' and any(s == n2 and r.startswith(':op') for s, r, t in amr.edges)]
                    for n2 in candidate_nodes[:]:
                        name_parts = [(r, t) for s, r, t in amr.edges if s == n2 and r.startswith(':op')]
                        name_parts = [t for r, t in sorted(name_parts, key=lambda x: int(x[0][3:]))]
                        label2 = '-'.join(amr.nodes[p] for p in name_parts).replace('"', '')
                        if label2!=label:
                            candidate_nodes.remove(n2)
                if len(candidate_nodes) != 1:
                    continue
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
                aligned_nodes.add(n)
                break
    # single token exact match
    for n in amr.nodes:
        if n in aligned_nodes: continue
        if amr.nodes[n].replace('"', '')[0].isalpha() and not amr.nodes[n].endswith('-91'):
            if is_name(n):
                name_node = [s for s, r, t in amr.edges if amr.nodes[s] == 'name' and t == n][0]
                name_parts = [(r, t) for s, r, t in amr.edges if s == name_node and r.startswith(':op')]
                name_parts = [t for r,t in sorted(name_parts, key=lambda x:int(x[0][3:]))]
                if len(name_parts) > 1 and len(node_label(amr, n))<=4:
                    continue
                if len(name_parts)>3:
                    continue
            candidate_spans = [span for span in amr.spans if any(amr.lemmas[t].lower() == node_label(amr, n).lower() for t in span)]
            candidate_spans = [span for span in candidate_spans if not any(node_label(amr, n2) == node_label(amr, n)
                                                                            for n2 in amr.get_alignment(alignments, token_id=span[0]).nodes)]
            candidate_nodes = [n2 for n2 in amr.nodes if node_label(amr, n) == node_label(amr, n2) and not amr.get_alignment(alignments, node_id=n2)]
            if len(candidate_spans) == 1 and len(candidate_nodes) == 1:
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
    if english:
        _exact_align_subgraphs_english(amr, alignments)



frame_re = re.compile('([a-z]|-)*-[0-9][0-9]')

def node_label(amr, n):
    node = amr.nodes[n].replace('"', '')
    if frame_re.match(node):
        node = '-'.join(node.split('-')[:-1])
    return node


time_re = re.compile('[0-2]?\d:[0-5]\d')


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
                    nums = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                            6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
                            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
                            16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
                            20: 'twenty', 30: 'thirty', 40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
                    if int(label) in nums:
                        candidate_strings.append(nums[int(label)])
                # Decades
                if amr.nodes[n].isdigit() and len(amr.nodes)==4:
                    years = {1920: ['twenties',"'20s",'20s'],
                             1930: ['thirties',"'30s",'30s'],
                             1940: ['forties',"'40s",'40s'],
                             1950: ['fifties',"'50s",'50s'],
                             1960: ['sixties',"'60s",'60s'],
                             1970: ['seventies',"'70s",'70s'],
                             1980: ['eighties',"'80s",'80s'],
                             1990: ['nineties',"'90s",'90s'],}
                    if int(label) in years:
                        candidate_strings.extend(years[int(label)])

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
                # Currency
                if any(amr.tokens[t] in ['$', '€', '£', '¥'] for t in align.tokens):
                    tok = [amr.tokens[t] for t in align.tokens if amr.tokens[t] in ['$', '€', '£', '¥']][0]
                    currency = {'$':'dollar', '€':'euro', '£':'pound', '¥':'yen'}
                    candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n2]==currency[tok]]
                    if len(candidate_nodes)==1:
                        align.nodes.append(candidate_nodes[0])

    # exact match special rules
    for n in amr.nodes:
        if amr.nodes[n] in ['i', 'we', 'you','it']:
            candidate_strings = {'i': ['i', 'my', 'mine', 'me'],
                                 'we': ['we', 'our', 'ours', 'us'],
                                 'you': ['you', 'your', 'yours'],
                                 'it': ['it', 'its', "it's"]}
            candidate_nodes = [n2 for n2 in amr.nodes if amr.nodes[n] == amr.nodes[n2]]
            if len(candidate_nodes) > 1 and n != candidate_nodes[0]: continue
            candidate_spans = [span for span in amr.spans if
                               len(span) == 1 and amr.lemmas[span[0]].lower() in candidate_strings[amr.nodes[n]]]
            if candidate_spans:
                for span in candidate_spans:
                    span_align = amr.get_alignment(alignments, token_id=span[0])
                    if span_align:
                        span_align.nodes = []
                span_align = amr.get_alignment(alignments, token_id=candidate_spans[0][0])
                span_align.nodes.append(n)
            continue
        align = amr.get_alignment(alignments, node_id=n)
        if not align:
            candidate_spans = []
            # exact match for 'and' and 'multi-sentence'
            if amr.nodes[n] in ['and', 'multi-sentence']:
                label = amr.nodes[n]
                candidate_spans = [span for span in amr.spans if not amr.get_alignment(alignments, token_id=span[0])]
                if label == 'and':
                    candidate_spans = [span for span in candidate_spans if
                                        ' '.join(amr.lemmas[t] for t in span).lower() in ['and', '&', 'additionally',
                                                                                          'as well', 'as well as', 'in addition']]
                    if not candidate_spans:
                        candidate_spans = [span for span in candidate_spans if
                                            ' '.join(amr.lemmas[t] for t in span).lower() in [',', ';']]
                elif label == 'multi-sentence':
                    candidate_spans = [span for span in candidate_spans if
                                        len(span) == 1 and amr.lemmas[span[0]] in ['.', ';']]
                    if not candidate_spans:
                        candidate_spans = [span for span in candidate_spans if
                                            len(span) == 1 and amr.lemmas[span[0]] in ['?', '!']]
                    if candidate_spans and amr.spans[-1] == candidate_spans[-1]:
                        candidate_spans.pop()
            # exact match for 'have-03'
            elif amr.nodes[n] == 'have-03':
                candidate_spans = [span for span in amr.spans if
                                    len(span) == 1 and amr.lemmas[span[0]].lower() in ['have', 'with', "'s"]]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for 'person'
            elif amr.nodes[n] == 'person':
                candidate_spans = [span for span in amr.spans if
                                    len(span) == 1 and amr.lemmas[span[0]].lower() in ['person', 'people']]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for 'include-91'
            elif amr.nodes[n] == 'include-91':
                candidate_spans = [span for span in amr.spans if
                                    ' '.join(amr.lemmas[t] for t in span).lower() in ['include', 'out of']]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for 'instead-of-91'
            elif amr.nodes[n] == 'instead-of-91':
                candidate_spans = [span for span in amr.spans if
                                   ' '.join(amr.lemmas[t] for t in span).lower() in ['instead', 'instead of']]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for 'cause-01'
            elif amr.nodes[n] == 'cause-01':
                candidate_spans = [span for span in amr.spans if
                                    ' '.join(amr.lemmas[t] for t in span).lower() in
                                    ['thus', 'since', 'because', 'cause', 'such', 'such that', 'so', 'therefore',
                                     'out of', 'due to', 'thanks to', 'reason', 'why', 'how', 'consequently', ',']]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for polarity -
            elif amr.nodes[n] == '-':
                candidate_spans = [span for span in amr.spans if
                                    ' '.join(amr.lemmas[t] for t in span).lower() in
                                    ['not', "n't", 'non', 'without', 'no', 'none', 'never', 'neither', 'no one']]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for amr-unknown
            elif amr.nodes[n] == 'amr-unknown':
                if any(r == ':polarity' and t == n for s, r, t in amr.edges):
                    candidate_spans = [span for span in amr.spans if
                                        ' '.join(amr.lemmas[t] for t in span).lower() in ['?']]
                    candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]

                else:
                    candidate_spans = [span for span in amr.spans if ' '.join(amr.lemmas[t] for t in span).lower() in
                                        ['why', 'how', 'when', 'where', 'who', 'which', 'what', 'how many', 'how long',
                                         'how much']]
                    candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for rate-entity-91
            elif amr.nodes[n] == 'rate-entity-91':
                candidate_spans = [span for span in amr.spans if
                                    ' '.join(amr.lemmas[t] for t in span).lower() in
                                    ['per', 'every', 'monthly', 'weekly', 'weekly', 'annually', 'annual', 'daily',
                                     'hourly']]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # exact match for mean-01
            elif amr.nodes[n] == 'mean-01':
                candidate_spans = [span for span in amr.spans if
                                    len(span) == 1 and amr.lemmas[span[0]].lower() in [':', ',', ]]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
            # United States
            elif amr.nodes[n] == 'name' and {amr.nodes[t].replace('"', '') for s, r, t in amr.edges if
                                             s == n and r.startswith(':op')} in [
                {'United', 'States'}, {'America'}, {'United', 'States', 'of', 'America'}]:
                candidate_spans = [span for span in amr.spans
                                    if not any(amr.nodes[n2].replace('"', '') in ['United', 'States', 'America']
                                               for n2 in amr.get_alignment(alignments, token_id=span[0]).nodes)]
                candidate_spans = [span for span in candidate_spans if any(amr.lemmas[t].replace('.', '') in
                                                                             ['American', 'US', 'USA'] for t in span)]
            candidate_nodes = [n2 for n2 in amr.nodes if
                               amr.nodes[n] == amr.nodes[n2] and not amr.get_alignment(alignments, node_id=n2)]
            if amr.nodes[n] == 'name':
                name = {amr.nodes[t].replace('"', '') for s, r, t in amr.edges if s == n and r.startswith(':op')}
                candidate_nodes = [n2 for n2 in candidate_nodes if
                                   {amr.nodes[t].replace('"', '') for s, r, t in amr.edges if
                                    s == n2 and r.startswith(':op')} == name]
            if len(candidate_spans) == 1 and len(candidate_nodes) == 1:
                span = candidate_spans[0]
                align = amr.get_alignment(alignments, token_id=span[0])
                align.nodes.append(n)
    for span in amr.spans:
        align = amr.get_alignment(alignments, token_id=span[0])
        if not align:
            candidate_spans = []
            candidate_nodes = []
            label = ' '.join(amr.lemmas[t] for t in span)
            # exact match for 'how'
            if label == 'how':
                candidate_spans = [s for s in amr.spans if len(s) == 1 and amr.lemmas[s[0]] == label]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
                candidate_nodes = []
                for n in amr.nodes:
                    if amr.nodes[n] == 'thing' and not amr.get_alignment(alignments, node_id=n):
                        if any((s == n and r == ':manner-of') or (t == 'n' and r == ':manner') for s, r, t in
                               amr.edges):
                            candidate_nodes.append(n)
                    elif amr.nodes[n] == 'so' and not amr.get_alignment(alignments, node_id=n):
                        so_tokens = [s for s in amr.spans if len(s) == 1 and amr.lemmas[s[0]] == 'so']
                        so_tokens = [s for s in so_tokens if not amr.get_alignment(alignments, token_id=s[0])]
                        if not so_tokens:
                            candidate_nodes.append(n)
                    elif amr.nodes[n] == 'have-manner-91' and not amr.get_alignment(alignments, node_id=n):
                        candidate_nodes.append(n)
                if len(candidate_spans) == 1 and len(candidate_nodes) == 1:
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(candidate_nodes[0])
            # as ... as construction
            elif label == 'as':
                candidate_spans = [s for s in amr.spans if len(s) == 1 and amr.lemmas[s[0]] == label]
                candidate_spans = [s for s in candidate_spans if not amr.get_alignment(alignments, token_id=s[0])]
                candidate_nodes = [n for n in amr.nodes if not amr.get_alignment(alignments, node_id=n)]
                candidate_nodes = [n for n in candidate_nodes if amr.nodes[n] in ['equal']]
                if len(candidate_spans) <= 2 and len(candidate_nodes) == 1 and span == candidate_spans[0]:
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(candidate_nodes[0])
                    continue
                if len(candidate_spans) == 1 and len(candidate_nodes) == 1:
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(candidate_nodes[0])
            # try un- non-
            elif len(span) == 1 and any(label.startswith(neg) for neg in ['un', 'non', 'in', 'im','il']):
                prefix = [pre for pre in ['un', 'non', 'in', 'im','il'] if label.startswith(pre)][0]
                candidate_spans = [span for span in amr.spans if
                                    len(span) == 1 and amr.lemmas[span[0]][:6] == label[:6]]
                candidate_spans = [span for span in candidate_spans if
                                    not amr.get_alignment(alignments, token_id=span[0])]
                candidate_nodes = []
                minus = None
                label = label[len(prefix):]
                for n in amr.nodes:
                    if amr.nodes[n].split('-')[0][:4] == label[:4]:
                        if amr.get_alignment(alignments, node_id=n): continue
                        m = [t for s, r, t in amr.edges if s == n and r == ':polarity' and amr.nodes[t] == '-']
                        if m and not amr.get_alignment(alignments, node_id=m[0]):
                            candidate_nodes.append(n)
                            minus = m[0]
                if len(candidate_spans) == 1 and len(candidate_nodes) == 1:
                    align = amr.get_alignment(alignments, token_id=span[0])
                    align.nodes.append(candidate_nodes[0])
                    align.nodes.append(minus)
            # WSJ Date format
            elif len(span) == 1 and label.isdigit() and (len(label) == 6 or (len(label)==8 and label[:2]=='20')):
                if len(label) == 8 and label[:2]=='20':
                    year = label[:4]
                    month = int(label[4:6])
                    day = int(label[6:])
                else:
                    year = int(label[:2])
                    if year < 20:
                        year = 2000 + year
                    else:
                        year = 1900 + year
                    month = int(label[2:4])
                    day = int(label[4:])
                for n in amr.nodes:
                    if amr.nodes[n] == 'date-entity':
                        if amr.get_alignment(alignments, node_id=n):
                            continue
                        year_node = [t for s, r, t in amr.edges if s == n and r == ':year']
                        month_node = [t for s, r, t in amr.edges if s == n and r == ':month']
                        day_node = [t for s, r, t in amr.edges if s == n and r == ':day']
                        if year_node and int(amr.nodes[year_node[0]]) == year:
                            if month == 0 or (month_node and int(amr.nodes[month_node[0]]) == month):
                                if day == 0 or (day_node and int(amr.nodes[day_node[0]]) == day):
                                    align.nodes.append(n)
                                    align.nodes.append(year_node[0])
                                    if month_node:
                                        align.nodes.append(month_node[0])
                                    if day_node:
                                        align.nodes.append(day_node[0])
                                    break


def english_is_alignment_forbidden(amr, span, n):

    token_label = ' '.join(amr.lemmas[t] for t in span)
    if amr.nodes[n] == 'person':
        if token_label not in ['person', 'people', 'those']:
            return True
    elif amr.nodes[n] == 'thing':
        if token_label not in ['thing', 'how']:
            return True
    elif amr.nodes[n] == 'multi-sentence':
        if len(span)!=1: return True
        if len(amr.tokens[span[0]]) != 1:
            return True
        if amr.tokens[span[0]].isalpha() or amr.tokens[span[0]].isdigit():
            return True
        if span[-1] == len(amr.tokens) - 1:
            return True
    elif amr.nodes[n] == 'and':
        semicolons = [span for span in amr.spans if ' '.join(amr.tokens[t] for t in span) == ';']
        if span in semicolons[1:]:
            return True
        if token_label.isalpha() and token_label not in ['and', 'as well', 'as well as', 'with', 'plus', 'additionally', 'in addition', 'addition',
                                                         'both', 'either', 'neither', 'nor', 'moreover', 'furthermore',]:
            return True
    elif amr.nodes[n] in ['more','have-degree-91','have-quant-91']:
        if token_label == 'than':
            return True
    if token_label == 'these' and amr.nodes[n] not in ['these','this']:
        return True
    elif token_label == 'those' and amr.nodes[n] not in ['that','those','person','thing']:
        return True
    elif token_label == 'this' and amr.nodes[n] not in ['this']:
        return True
    elif token_label == 'whose' and amr.nodes[n] not in ['person','own-01','amr-unknown','have-03']:
        return True
    elif token_label == 'who' and amr.nodes[n] not in ['person','amr-unknown']:
        return True
    elif token_label == 'which' and amr.nodes[n] not in ['amr-unknown','thing']:
        return True
    elif token_label == 'have' and not amr.nodes[n].startswith('have'):
        return True
    elif token_label == 'be' and amr.nodes[n] not in ['be-02','exist-01']:
        return True
    elif token_label == 'will' and not amr.nodes[n].startswith('will'):
        return True
    return False


def clean_subgraph(amr, alignments, align):

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
            largest_align = max(components, key=lambda a:len(a.nodes))
            return largest_align
    if not align.tokens:
        raise Exception('Alignment Error, Missing Tokens:', amr.id, str(align))
    return align


def clean_alignments(amr, alignments):
    node_occurence = Counter()
    for align2 in alignments[amr.id]:
        # if align2.type.startswith('dupl'): continue
        for n in align2.nodes:
            node_occurence[n] += 1
            if n not in amr.nodes:
                raise Exception('Alignment Error, Unrecognized Node:', amr.id, n)
    if any(node_occurence[n]>1 for n in node_occurence):
        ns = [n2 for n2 in node_occurence if node_occurence[n2]>1]
        for n in ns:
            aligns = [align2 for align2 in alignments[amr.id] if n in align2.nodes]
            largest_align = max(aligns, key=lambda a:len(a.nodes))
            for align2 in aligns:
                align2.nodes = [n2 for n2 in align2.nodes if n2!=n]
            largest_align.nodes.append(n)
    for align2 in alignments[amr.id][:]:
        if align2.type.startswith('dupl') and not align2.nodes:
            alignments[amr.id].remove(align2)


def separate_components(amr, align):
    node_labels = [amr.nodes[n] for n in align.nodes]
    if len(node_labels) > 1 and all(node == node_labels[0] for node in node_labels):
        return [AMR_Alignment(type='subgraph', tokens=align.tokens, nodes=[n], amr=amr) for n in align.nodes]
    if not align.nodes:
        return [align]
    if is_subgraph(amr, align.nodes):
        return [align]
    components = get_connected_components(amr, align.nodes)
    components = [list(sub.nodes.keys()) for sub in components]
    components = [AMR_Alignment(type='subgraph', tokens=align.tokens, nodes=nodes, amr=amr) for nodes in components]
    return components


def is_subgraph(amr, nodes):
    if is_rooted_dag(amr, nodes):
        return True
    # handle "never => ever, -" and other similar cases
    if len(nodes) == 2:
        nodes = nodes.copy()
        parents1 = [s for s, r, t in amr.edges if t == nodes[0]]
        parents2 = [s for s, r, t in amr.edges if t == nodes[1]]
        children = [t for s, r, t in amr.edges if s in nodes]
        # rels = [r for s, r, t in amr.edges if t in nodes]
        if amr.nodes[nodes[0]] == amr.nodes[nodes[1]]:
            return False
        if parents1 == parents2 and len(parents1) == 1 and not children:
            return True
    return False
