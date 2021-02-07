from amr_utils.amr_readers import JAMR_AMR_Reader


def main():
    file = 'data/szubert/szubert_amrs.txt'
    file2 = 'data/ldc_train.txt'

    output1 = 'data/szubert/szubert_sents.isi.txt'
    output2 = 'data/szubert/szubert_amrs.isi.txt'
    output3 = 'data/szubert/szubert_ids.isi.txt'

    cr = JAMR_AMR_Reader()
    amrs = cr.load(file, remove_wiki=True)
    amrs += cr.load(file2, remove_wiki=True)
    unique_ids = set()
    amrs2 = []
    for amr in amrs:
        if amr.id not in unique_ids:
            unique_ids.add(amr.id)
            amrs2.append(amr)
    amrs = amrs2

    with open(output1, 'w+', encoding='utf8') as f:
        for amr in amrs:
            tokens = [t for t in amr.tokens]
            for i, t in enumerate(tokens):
                if t[0] == '@' and t[-1] == '@' and len(t) == 3:
                    tokens[i] = t[1]
            f.write(' '.join(tokens) + '\n')
    with open(output2, 'w+', encoding='utf8') as f:
        for amr in amrs:
            graph_string = amr.graph_string()\
                .replace('/', ' / ')\
                .replace('\n','')\
                .replace('\r','') \
                .replace('\t:', ' :') \
                .replace('\t','')
            f.write(graph_string+'\n')
    with open(output3, 'w+', encoding='utf8') as f:
        for amr in amrs:
            f.write(amr.id+'\n')


if __name__ == '__main__':
    main()