import json
import sys

from amr_utils.amr_readers import AMR_Reader

import stanza
from spacy.tokens.doc import Doc
from tqdm import tqdm



class NoTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        spaces = [True] * len(tokens)
        return Doc(self.vocab, words=tokens, spaces=spaces)

def main():
    amr_file = sys.argv[1]
    # output_file = sys.argv[2]

    # stanza.download('en')
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)

    lemmas_json = {}
    pos_json = {}

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

        lemmas = ['' for _ in amr.tokens]
        pos = ['' for _ in amr.tokens]
        for i in stanza_lemmas:
            lemmas[convert_ids[i]] += stanza_lemmas[i]
            pos[convert_ids[i]] = stanza_pos[i]
        for i, l in enumerate(lemmas):
            if not l and i > 0:
                lemmas[i] = lemmas[i - 1]
                pos[i] = pos[i - 1]

        lemmas_json[amr.id] = lemmas
        pos_json[amr.id] = pos

    filename = amr_file.replace('.txt', '')
    with open(filename + '.lemmas.json', 'w+', encoding='utf8') as f:
        json.dump(lemmas_json, f)
    with open(filename + '.pos.json', 'w+', encoding='utf8') as f:
        json.dump(pos_json, f)


if __name__ == '__main__':
    main()
