import stanza
from tqdm import tqdm


def main():
    file = '../data/ldc_train.ccg.txt'
    file2 = '../data/ldc_train.supertags.txt'

    nlp = stanza.Pipeline('en', processors='tokenize,pos')

    sents = []
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            row = line.split()
            sent = []
            tokens = []
            for word in row:
                index = len(word) - 1 - word[::-1].index('|')
                word = [word[:index], word[index+1:]]
                sent.append(word)
                tokens.append(word[0])
            sents.append(sent)
            doc = nlp(' '.join(tokens))
            word_boundaries = [0]
            total = 0
            for tok in tokens:
                total+=len(tok)+1
                word_boundaries.append(total)
            sent_pos = []
            i = 0
            for s in doc.sentences:
                for token in s.tokens:
                    start = token.start_char
                    end = token.end_char
                    for word in token.words:
                        pos = word.xpos
                        if start<word_boundaries[i]:
                            continue
                        while end>word_boundaries[i+1]:
                            sent_pos.append(pos)
                            i+=1
                        sent_pos.append(pos)
                        i+=1
            if len(sent_pos)!=len(tokens):
                raise Exception()
            for word, pos in zip(sent, sent_pos):
                word.append(pos)

    with open(file2, 'w+', encoding='utf8') as f:
        for sent in sents:
            for word, tag, pos in sent:
                f.write('\t'.join([word, pos, '1', tag, '1'])+'\n')
            f.write('\n')





if __name__ == '__main__':
    main()