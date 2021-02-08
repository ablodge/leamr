import sys

from amr_utils.amr_readers import AMR_Reader

from display import Display
from evaluate.utils import evaluate, perplexity
from models.subgraph_model import Subgraph_Model
from nlp_data import add_nlp_data


def report_progress(amrs, amr_file, alignments, reader, epoch=None):
    epoch = '' if epoch is None else f'.epoch{epoch}'
    Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.subgraph_alignments{epoch}.html')

    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments{epoch}.json'
    print(f'Writing subgraph alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, alignments)

def get_eval_data(reader):
    if len(sys.argv) > 2:
        eval_amr_file = sys.argv[2]
        eval_amrs = reader.load(eval_amr_file, remove_wiki=True)
        add_nlp_data(eval_amrs, eval_amr_file)
        gold_dev_alignments = reader.load_alignments_from_json(sys.argv[3], eval_amrs)
        return eval_amr_file, eval_amrs, gold_dev_alignments
    return None, None, None


def main():

    amr_file = sys.argv[1]

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    # amrs = amrs[:100]
    add_nlp_data(amrs, amr_file)

    eval_amr_file, eval_amrs, gold_dev_alignments = get_eval_data(cr)
    eval_amr_ids = {amr.id for amr in eval_amrs}
    amrs = [amr for amr in amrs if amr.id not in eval_amr_ids]

    align_model = Subgraph_Model(amrs, align_duplicates=True)

    iters = 3

    alignments = None
    eval_alignments = None

    for i in range(iters):
        print(f'Epoch {i}: Training data')
        alignments = align_model.align_all(amrs)
        align_model.update_parameters(amrs, alignments)
        report_progress(amrs, amr_file, alignments, reader, epoch=i)
        print()

        if eval_amrs:
            print(f'Epoch {i}: Evaluation data')
            eval_alignments = align_model.align_all(eval_amrs)
            perplexity(align_model, eval_amrs, eval_alignments)
            evaluate(eval_amrs, eval_alignments, gold_dev_alignments)
            print()

    report_progress(amrs, amr_file, alignments, reader)

    if eval_amrs:
        report_progress(eval_amrs, eval_amr_file, eval_alignments, reader)

if __name__=='__main__':
    main()