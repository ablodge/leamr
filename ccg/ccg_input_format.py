from amr_utils.amr_readers import AMR_Reader

from nlp_data import add_nlp_data


def main():
    file = '../data/split/train.txt'
    file2 = '../data/train.sents.txt'

    reader = AMR_Reader()
    amrs = reader.load(file, remove_wiki=True)
    add_nlp_data(amrs, file)

    # amrs2 = reader.load('../data/split/test.txt', remove_wiki=True)
    # add_nlp_data(amrs2, '../data/split/test.txt')
    # amrs = amrs+amrs2

    with open(file2, 'w+', encoding='utf8') as f:
        for amr in amrs:
            for token, pos in zip(amr.tokens, amr.pos):
                f.write(f'{token}|{pos} ')
            f.write('\n')


if __name__ == '__main__':
    main()