import sys

from amr_utils.amr_readers import AMR_Reader

from display import Alignment_Display
from evaluate.utils import perplexity, evaluate_relations
from models.relation_model import Relation_Model
from nlp_data import add_nlp_data


def report_progress(amrs, amr_file, alignments, reader, epoch=None):
    epoch = '' if epoch is None else f'.epoch{epoch}'
    Alignment_Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.relation_alignments{epoch}.html', alignments)

    align_file = amr_file.replace('.txt', '') + f'.relation_alignments{epoch}.json'
    print(f'Writing relation alignments to: {align_file}')
    reader.save_alignments_to_json(align_file, alignments)


def get_eval_data(reader):
    if len(sys.argv) > 2:
        eval_amr_file = sys.argv[2]
        eval_amrs = reader.load(eval_amr_file, remove_wiki=True)
        add_nlp_data(eval_amrs, eval_amr_file)
        gold_eval_alignments = reader.load_alignments_from_json(sys.argv[3], eval_amrs) if len(sys.argv)>3 else None
        return eval_amr_file, eval_amrs, gold_eval_alignments
    return None, None, None


def main():
    amr_file = sys.argv[1]

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    add_nlp_data(amrs, amr_file)

    eval_amr_file, eval_amrs, gold_eval_alignments = get_eval_data(reader)
    eval_amr_ids = {amr.id for amr in eval_amrs}
    amrs = [amr for amr in amrs if amr.id not in eval_amr_ids]
    # amrs = amrs[:1000]

    align_file = amr_file.replace('.txt', '') + '.subgraph_alignments.json'
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)

    if gold_eval_alignments is not None:
        align_file = eval_amr_file.replace('.txt', '') + '.subgraph_alignments.gold.json'
        gold_subgraph_alignments = reader.load_alignments_from_json(align_file, eval_amrs)
        align_file = eval_amr_file.replace('.txt', '') + '.subgraph_alignments.json'
        pred_subgraph_alignments = reader.load_alignments_from_json(align_file, eval_amrs)
        # pred_subgraph_alignments = gold_subgraph_alignments
        for amr_id in pred_subgraph_alignments:
            subgraph_alignments[amr_id] = pred_subgraph_alignments[amr_id]
        for amr in eval_amrs:
            spans = [align.tokens for align in pred_subgraph_alignments[amr.id] if align.type=='subgraph']
            amr.spans = spans

    align_model = Relation_Model(amrs, subgraph_alignments)

    iters = 3

    alignments = None
    eval_alignments = None

    for i in range(iters):
        print(f'Epoch {i}: Training data')
        alignments = align_model.align_all(amrs)
        align_model.update_parameters(amrs, alignments)
        report_progress(amrs, amr_file, alignments, reader, epoch=i)
        perplexity(align_model, amrs, alignments)
        print()

        if eval_amrs:
            print(f'Epoch {i}: Evaluation data')
            eval_alignments = align_model.align_all(eval_amrs)
            perplexity(align_model, eval_amrs, eval_alignments)
            if gold_eval_alignments is not None:
                evaluate_relations(eval_amrs, eval_alignments, gold_eval_alignments, pred_subgraph_alignments, gold_subgraph_alignments)
                # evaluate(eval_amrs, eval_alignments, gold_eval_alignments, mode='edges')
            print()

    report_progress(amrs, amr_file, alignments, reader)

    if eval_amrs:
        report_progress(eval_amrs, eval_amr_file, eval_alignments, reader)


if __name__ == '__main__':
    main()