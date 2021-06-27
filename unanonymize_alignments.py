import os

from amr_utils.alignments import load_from_json, write_to_json
from amr_utils.amr_readers import AMR_Reader


def main():
    dir = 'data-release/alignments'

    reader = AMR_Reader()
    dev_amrs = reader.load('data-release/amrs/leamr_dev.txt')
    test_amrs = reader.load('data-release/amrs/leamr_test.txt')
    all_amrs = reader.load('data-release/amrs/ldc+little_prince.txt')

    amr_map = {'leamr_dev':dev_amrs, 'leamr_test':test_amrs, 'ldc+little_prince':all_amrs}
    for filename in os.listdir(dir):
        file = os.path.join(dir, filename)
        if file.endswith('alignments.json') or file.endswith('alignments.gold.json'):
            for k in amr_map:
                if filename.startswith(k):
                    amrs = amr_map[k]
                    aligns = load_from_json(file, amrs, unanonymize=True)
                    # run quick test
                    for amr in amrs:
                        for align in aligns[amr.id]:
                            for n in align.nodes:
                                if n not in amr.nodes:
                                    raise Exception(f'Failed to match alignments to AMR data. AMR "{amr.id}" has no node named "{n}".')
                    # write output
                    write_to_json(file, aligns, amrs=amrs, anonymize=False)
                    break

if __name__=='__main__':
    main()