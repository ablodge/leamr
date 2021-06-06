import json
import sys

import spacy
from amr_utils.amr_readers import AMR_Reader

import stanza
import neuralcoref
from spacy.tokens.doc import Doc
from tqdm import tqdm

from rule_based import mwes

SEE_GOLD_AMR = True


def add_nlp_data(amrs, file):
    lemmas_file = file.replace('.txt', '') + '.lemmas.json'
    with open(lemmas_file, 'r') as f:
        lemmas = json.load(f)
    span_file1 = file.replace('.txt', '') + '.spans.json'
    with open(span_file1, 'r') as f:
        spans = json.load(f)
    coref_file = file.replace('.txt', '') + '.coref.json'
    with open(coref_file, 'r') as f:
        corefs = json.load(f)
    pos_file = file.replace('.txt', '') + '.pos.json'
    with open(pos_file, 'r') as f:
        pos = json.load(f)
    for amr in amrs:
        amr_id = amr.id
        if amr_id.endswith('#2'):
            amr_id = amr_id.split('#')[0]
        amr.lemmas = lemmas[amr_id]
        amr.spans = spans[amr_id]
        amr.coref = corefs[amr_id]
        amr.pos = pos[amr.id]


def get_mwe_types_by_first_token():
    hyphenated = [' - '.join(mwe.split()) for mwe in mwes.OTHER_MWES]
    all_mwe_types = set(mwes.PMWES + mwes.VMWES + mwes.OTHER_MWES + mwes.HAND_ADDED_MWES + hyphenated)
    all_mwe_types = [tuple(mwe.split()) for mwe in all_mwe_types]
    all_mwe_types = [mwe for mwe in sorted(all_mwe_types, key=lambda x: len(x), reverse=True)]
    all_mwe_types_dict = {}
    for mwe in all_mwe_types:
        first = mwe[0]
        if first not in all_mwe_types_dict:
            all_mwe_types_dict[first] = []
        all_mwe_types_dict[first].append(mwe)

    return all_mwe_types_dict


class NoTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        spaces = [True] * len(tokens)
        return Doc(self.vocab, words=tokens, spaces=spaces)


def get_root(amr, token_ids):
    alignments = [amr.get_alignment(token_id=t) for t in token_ids]
    nodes = {n for align in alignments for n in align.nodes}
    if nodes:
        align_amr = amr.get_subgraph(list(nodes))
        n = align_amr.root
    else:
        n = None
    return n


def get_coref_parser():
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    nlp.tokenizer = NoTokenizer(nlp.vocab)
    return nlp


def get_corefs(amr, parser):
    nlp = parser(amr.tokens)
    coref_list = []
    for ent in nlp._.coref_clusters:
        spans = []
        for s in ent.mentions:
            mention_span = [i for i in range(s.start, s.end)]

            if len(mention_span) > 1:
                if amr.tokens[mention_span[0]] in ['the']:
                    mention_span = mention_span[1:]
            spans.append(mention_span)
        coref_list.append(spans)

    return coref_list


def main():
    amr_file = sys.argv[1]
    # output_file = sys.argv[2]

    # stanza.download('en')
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)

    lemmas_json = {}
    pos_json = {}
    ner_spans = {}
    mwe_spans = {}
    multi_word_spans = {}
    coreferences = {}

    mwe_types = get_mwe_types_by_first_token()
    coref_parser = None
    try:
        coref_parser = get_coref_parser()
    except Exception as e:
        print('Warning: Failed to parse coreference. '
              'Please install neuralcoref from source: https://github.com/huggingface/neuralcoref#install-neuralcoref-from-source',
              file=sys.stderr)

    enum_amrs = [_ for _ in enumerate(amrs)]
    for amr_idx, amr in tqdm(enum_amrs):
        tokens = amr.tokens.copy()
        for i, tok in enumerate(tokens):
            if tok.startswith('@') and tok.endswith('@') and len(tok) == 3:
                tokens[i] = tok[1]
        doc = nlp(' '.join(tokens))
        start_idx = {}
        end_idx = {}
        i = 0
        for j, tok in enumerate(tokens):
            start_idx[j] = i
            end_idx[j] = i + len(tok)
            i += len(tok) + 1

        convert_ids = {}
        stanza_lemmas = {}
        stanza_entity_type = []
        stanza_entity_spans = []
        stanza_pos = {}
        for s in doc.sentences:
            for token in s.tokens:
                start = token.start_char
                end = token.end_char
                idx = [k for k in start_idx if start >= start_idx[k] and end <= end_idx[k]]
                if len(idx) == 0:
                    idx = [k for k in start_idx if start <= start_idx[k] <= end]
                idx = idx[0]
                convert_ids[start] = idx
                for word in token.words:
                    if start not in stanza_lemmas:
                        stanza_lemmas[start] = ''
                    lemma = word.lemma
                    stanza_lemmas[start] += lemma
                    stanza_pos[start] = word.xpos
            for e in s.entities:
                stanza_entity_type.append(e.type)
                ent_type = e.type
                span = []
                for t in e.tokens:
                    start = t.start_char
                    span.append(start)
                # name = ' '.join(amr.tokens[convert_ids[t]] for t in span)
                # type = e.type
                pos = [stanza_pos[t] for t in span]
                if pos[0] in ['DT', 'PDT', 'PRP$', 'RB', 'RP', 'JJ', 'JJR', 'JJS', 'IN']:
                    while pos and pos[0] in ['DT', 'PDT', 'PRP$', 'RB', 'RP', 'JJ', 'JJR', 'JJS', 'IN']:
                        pos = pos[1:]
                        span = span[1:]
                    if len(span) == 0:
                        stanza_entity_type.pop()
                        continue
                if pos and pos[-1] in ['POS', 'RB', 'RBR', 'RBS']:
                    span = span[:-1]
                    if len(span) == 0:
                        stanza_entity_type.pop()
                        continue
                # next_tok = convert_ids[span[-1]]
                # next_tok = [s for s,t in convert_ids.items() if t==next_tok+1]
                # prev_tok = convert_ids[span[0]]
                # prev_tok = [s for s, t in convert_ids.items() if t == prev_tok - 1]
                # if next_tok:
                #     next_tok = next_tok[0]
                #     next_pos = stanza_pos[next_tok]
                #     if next_pos == 'NNP':
                #         span.append(next_tok)
                # if prev_tok:
                #     prev_tok = prev_tok[0]
                #     prev_pos = stanza_pos[prev_tok]
                #     if prev_pos == 'NNP':
                #         span.insert(0,prev_tok)
                # if amr.id=='bolt12_10494_3592.5':
                #     print()
                if len(span) == 1:
                    stanza_entity_type.pop()
                    continue
                stanza_entity_spans.append(span)
                # if ent_type in ['DATE','TIME','MONEY','QUANTITY']:
                #     print()
                #     print(ent_type, ' '.join(amr.tokens[convert_ids[i]] for i in span))
                #     print()

        lemmas = ['' for _ in amr.tokens]
        pos = ['' for _ in amr.tokens]
        for i in stanza_lemmas:
            lemmas[convert_ids[i]] += stanza_lemmas[i]
            pos[convert_ids[i]] = stanza_pos[i]
        for i, l in enumerate(lemmas):
            if not l and i > 0:
                lemmas[i] = lemmas[i - 1]
                pos[i] = pos[i - 1]
        entities = []
        for span in stanza_entity_spans:
            span = [convert_ids[i] for i in span]
            start = min(span)
            end = max(span) + 1
            entities.append((start, end))
        lemmas_json[amr.id] = lemmas
        pos_json[amr.id] = pos
        ner_spans[amr.id] = entities

        # get MWE spans
        mwe_spans[amr.id] = []
        taken = []
        for i, token in enumerate(amr.tokens):
            if i in taken: continue
            found = False
            token = token.lower()
            lemma = lemmas[i].lower()
            if token in mwe_types:
                for mwe in mwe_types[token]:
                    size = len(mwe)
                    if i + size - 1 >= len(amr.tokens): continue
                    if all(amr.tokens[i + idx].lower().replace('@', '') == mwe[idx] for idx in range(size)):
                        span = (i, i + size)
                        mwe_spans[amr.id].append(span)
                        for t in range(span[0], span[-1]):
                            taken.append(t)
                        found = True
                        break
            if found: continue
            if lemma in mwe_types:
                for mwe in mwe_types[lemma]:
                    size = len(mwe)
                    if i + size - 1 >= len(amr.tokens): continue
                    if all(lemmas[i + idx].lower().replace('@', '') == mwe[idx] for idx in range(size)):
                        span = (i, i + size)
                        mwe_spans[amr.id].append(span)
                        for t in range(span[0], span[-1]):
                            taken.append(t)
                        break
            taken.append(i)

        # look for names matching gold amr
        name_spans = []
        if SEE_GOLD_AMR:
            for n in amr.nodes:
                if amr.nodes[n] == 'name':
                    parts = [(int(r[3:]), t) for s, r, t in amr.edges if s == n and r.startswith(':op')]
                    parts = [t for r, t in sorted(parts, key=lambda x: x[0])]
                    label = ' '.join(amr.nodes[t].replace('"', '') for t in parts)
                    name_type = [s for s, r, t in amr.edges if t == n and r == ':name']
                    name_type = amr.nodes[name_type[0]] if name_type else None
                    if parts:
                        for start in range(len(amr.tokens)):
                            span = [t for t in range(start, start + len(parts))]
                            if span[-1] >= len(amr.tokens): break
                            tokens = [amr.tokens[t] for t in span]
                            token_label = ' '.join([tok for tok in tokens if tok != '"'])
                            if token_label.lower() == label.lower():
                                next_tok = span[-1] + 1
                                if next_tok < len(amr.tokens) and amr.tokens[next_tok] == name_type:
                                    span += [next_tok]
                                if len(parts) > 1:
                                    name_spans.append((span[0], span[-1] + 1))
                                start = span[0]
                                end = span[-1] + 1
                                for span in ner_spans[amr.id][:]:
                                    if span[0] <= start < span[1] and span[0] < end <= span[1] and (start, end) != span:
                                        ner_spans[amr.id].remove(span)
                                        break
                                break
            for t in range(len(amr.tokens)):
                if t + 2 < len(amr.tokens) and amr.tokens[t + 1] == '@-@':
                    label1 = f'{lemmas[t]}{lemmas[t + 2]}'.lower()[:len(lemmas[t]) + 4]
                    label2 = f'{lemmas[t]}-{lemmas[t + 2]}'.lower()[:len(lemmas[t]) + 5]
                    if any(amr.nodes[n].startswith(label1) or amr.nodes[n].startswith(label2) for n in amr.nodes):
                        name_spans.append((t, t + 3))
        # times
        taken = set()
        for t in range(len(amr.tokens)):
            if t in taken: continue
            start = t
            if amr.tokens[t].isdigit() and len(amr.tokens[t]) <= 2 and t + 2 < len(amr.tokens):
                if amr.tokens[t + 1] in ['@:@', ':'] and amr.tokens[t + 2].isdigit() and len(amr.tokens[t + 2]) == 2:
                    end = t + 2
                    while end + 1 < len(amr.tokens) \
                            and (amr.tokens[end + 1] in ['am', 'pm', 'a.m.', 'p.m.', '@:@', ':', 'UTC', 'GMT', 'EST',
                                                         'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                                         'Friday', 'Saturday', ]
                                 or (amr.tokens[end] in ['@:@', ':'] and amr.tokens[end + 1].isdigit() and len(
                                amr.tokens[end + 1]) == 2)):
                        end += 1
                    end += 1
                    time = ' '.join(amr.tokens[t] for t in range(start, end))
                    name_spans.append((start, end))
                    for span in ner_spans[amr.id]:
                        if start < span[1] < end and span[0] < start:
                            name_spans[-1] = (span[0], end)
                            break
                        elif start < span[0] < end and span[1] > end:
                            name_spans[-1] = (start, span[1])
                            break
                        elif span[0] <= start < span[1] and span[0] < end <= span[1]:
                            name_spans[-1] = span
                            break
                    start, end = name_spans[-1]
                    for i in range(start, end):
                        taken.add(i)
        multi_word_spans[amr.id] = []
        taken = set()
        for i, tok in enumerate(amr.tokens):
            if i in taken: continue
            if any(i == span[0] for span in name_spans):
                span = [s for s in name_spans if s[0] <= i < s[1]][0]
                span = [i for i in range(span[0], span[1])]
                multi_word_spans[amr.id].append(span)
                taken.update(span)
            elif any(i == span[0] for span in ner_spans[amr.id]):
                span = [s for s in ner_spans[amr.id] if s[0] <= i < s[1]][0]
                span = [i for i in range(span[0], span[1])]
                multi_word_spans[amr.id].append(span)
                taken.update(span)
            elif any(i == span[0] for span in mwe_spans[amr.id]):
                span = [s for s in mwe_spans[amr.id] if s[0] <= i < s[1]][0]
                span = [i for i in range(span[0], span[1])]
                multi_word_spans[amr.id].append(span)
                taken.update(span)
            else:
                multi_word_spans[amr.id].append([i])
                taken.add(i)
        if coref_parser is not None:
            corefs = get_corefs(amr, coref_parser)
            coreferences[amr.id] = corefs

    # ner_spans = {k: v for k, v in ner_spans.items() if v}
    # mwe_spans = {k: v for k, v in mwe_spans.items() if v}

    filename = amr_file.replace('.txt', '')
    with open(filename + '.lemmas.json', 'w+', encoding='utf8') as f:
        json.dump(lemmas_json, f)
    with open(filename + '.pos.json', 'w+', encoding='utf8') as f:
        json.dump(pos_json, f)
    with open(filename + '.spans.json', 'w+', encoding='utf8') as f:
        json.dump(multi_word_spans, f)
    if coreferences:
        with open(filename + '.coref.json', 'w+', encoding='utf8') as f:
            json.dump(coreferences, f)

    # for amr in amrs:
    #     print(' '.join('_'.join(amr.tokens[t] for t in span) for span in multi_word_spans[amr.id]))
    #     print()


if __name__ == '__main__':
    main()
