from pathlib import Path
from random import shuffle

from amr_utils.amr_readers import AMR_Reader


def main():

    split_file = Path('../data/split/train_ids.txt')
    if split_file.is_file():
        raise Exception('Cannot create Train, Dev, Test split because split already exists:',
                        str(split_file.resolve()))

    reader = AMR_Reader()

    ldc_train = '../data/ldc_train.txt'
    ldc_train = reader.load(ldc_train, remove_wiki=True)
    ldc_dev = '../data/ldc_dev.txt'
    ldc_dev = reader.load(ldc_dev, remove_wiki=True)
    ldc_test = '../data/ldc_test.txt'
    ldc_test = reader.load(ldc_test, remove_wiki=True)

    little_prince = '../data/little_prince.txt'
    little_prince = reader.load(little_prince, remove_wiki=True)
    szubert = '../data/szubert/szubert_amrs.txt'
    szubert = reader.load(szubert, remove_wiki=True)
    gold_dev = '../data/gold_dev/ldc_dev.gold.txt'
    gold_dev = reader.load(gold_dev, remove_wiki=True)

    szubert_ids = [amr.id for amr in szubert]
    train_ids = [amr.id for amr in ldc_train if amr.id not in szubert_ids]
    little_prince_ids = [amr.id for amr in little_prince if amr.id not in szubert_ids]
    gold_dev_ids = [amr.id for amr in gold_dev if amr.id not in szubert_ids]

    shuffle(little_prince_ids)
    sample = little_prince_ids[:50]


    new_train_ids = train_ids + [n for n in little_prince_ids if n not in sample]
    new_dev_ids = gold_dev_ids + sample
    new_test_ids = szubert_ids

    little_prince1 = [amr.id for amr in little_prince if amr.id in new_train_ids]
    little_prince2 = [amr.id for amr in little_prince if amr.id in new_dev_ids]
    little_prince3 = [amr.id for amr in little_prince if amr.id in new_test_ids]
    print('Split up little prince:', len(little_prince1), len(little_prince2), len(little_prince3))

    with open('../data/split/train_ids.txt', 'w+') as f:
        f.write(f'# {len(new_train_ids)} train AMRs\n')
        for n in sorted(new_train_ids):
            f.write(n+'\n')

    with open('../data/split/dev_ids.txt', 'w+') as f:
        f.write(f'# {len(new_dev_ids)} dev AMRs\n')
        for n in sorted(new_dev_ids):
            f.write(n + '\n')

    with open('../data/split/test_ids.txt', 'w+') as f:
        f.write(f'# {len(new_test_ids)} test AMRs\n')
        for n in sorted(new_test_ids):
            f.write(n + '\n')

    train_amrs = {amr.id:amr for amr in ldc_train+little_prince}
    reader.write_to_file('../data/split/train.txt', [train_amrs[n] for n in sorted(new_train_ids)])
    dev_amrs = {amr.id: amr for amr in ldc_dev + little_prince}
    reader.write_to_file('../data/split/dev.txt', [dev_amrs[n] for n in sorted(new_dev_ids)])
    test_amrs = {amr.id: amr for amr in szubert}
    reader.write_to_file('../data/split/test.txt', [test_amrs[n] for n in sorted(new_test_ids)])







if __name__=='__main__':
    main()