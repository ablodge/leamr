from amr_utils.amr_readers import AMR_Reader


def main():
    file = '../data/little_prince.txt'
    file2 = '../data/little_prince.sents.txt'

    reader = AMR_Reader()
    amrs = reader.load(file, remove_wiki=True)

    with open(file2, 'w+', encoding='utf8') as f:
        for amr in amrs:
            f.write(' '.join(amr.tokens)+'\n')


if __name__ == '__main__':
    main()