import json
import os
import sys

from amr_utils.amr_readers import AMR_Reader


def main():
    ldc_amrs_dir = sys.argv[1]
    lpp_amrs_file = 'data-release/amrs/little_prince.txt'
    additional_amrs_file = 'data-release/amrs/additional_amrs.txt'

    dev_ids = 'data-release/leamr_dev_ids.txt'
    with open(dev_ids) as f:
        dev_ids = [line.strip() for line in f if not line.strip().startswith('#')]
    test_ids = 'data-release/leamr_test_ids.txt'
    with open(test_ids) as f:
        test_ids = [line.strip() for line in f if not line.strip().startswith('#')]

    ldc_amrs_dir_train = os.path.join(ldc_amrs_dir, 'data/alignments/split', 'training')
    ldc_amrs_dir_dev = os.path.join(ldc_amrs_dir, 'data/alignments/split', 'dev')
    ldc_amrs_dir_test = os.path.join(ldc_amrs_dir, 'data/alignments/split', 'test')

    reader = AMR_Reader()
    ldc_amrs_train = reader.load_from_dir(ldc_amrs_dir_train)
    ldc_amrs_dev = reader.load_from_dir(ldc_amrs_dir_dev)
    ldc_amrs_test = reader.load_from_dir(ldc_amrs_dir_test)
    lpp_amrs = reader.load(lpp_amrs_file)
    add_amrs = reader.load(additional_amrs_file)

    all_amrs = ldc_amrs_train + ldc_amrs_dev + ldc_amrs_test + lpp_amrs+add_amrs
    all_amrs = {amr.id:amr for amr in all_amrs}
    dev_amrs = [all_amrs[amr_id] for amr_id in dev_ids]
    test_amrs = [all_amrs[amr_id] for amr_id in test_ids]

    print()

    output_file = 'data-release/amrs/ldc_train.txt'
    print('Writing LDC training AMRs to:', output_file)
    reader.write_to_file(output_file, ldc_amrs_train)

    output_file = 'data-release/amrs/ldc_dev.txt'
    print('Writing LDC development AMRs to:', output_file)
    reader.write_to_file(output_file, ldc_amrs_dev)

    output_file = 'data-release/amrs/ldc_test.txt'
    print('Writing LDC test AMRs to:', output_file)
    reader.write_to_file(output_file, ldc_amrs_test)

    output_file = 'data-release/amrs/leamr_dev.txt'
    print('Writing LEAMR development data to:', output_file)
    reader.write_to_file(output_file, dev_amrs)

    output_file = 'data-release/amrs/leamr_test.txt'
    print('Writing LEAMR test data to:', output_file)
    reader.write_to_file(output_file, test_amrs)

    output_file = 'data-release/amrs/ldc+little_prince.txt'
    print('Writing LDC + Little Prince data to:', output_file)
    all_amrs = ldc_amrs_train + ldc_amrs_dev + ldc_amrs_test + lpp_amrs
    reader.write_to_file(output_file, all_amrs)


if __name__ == '__main__':
    main()