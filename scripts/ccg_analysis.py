import re
import sys
from collections import Counter

from amr_utils.amr_readers import AMR_Reader

from nlp_data import add_nlp_data

from tree_st.allennlp.field import cat_as_json
from tree_st.util.reader import CategoryReader

cr = CategoryReader()

def get_ccg_tag(tag):
    try:
        ccg = cat_as_json(cr.read(tag))
    except:
        return {'root':'-ERR-','category':'-ERR-'}
    clean(ccg)
    depth_first = [ccg]
    while depth_first:
        dict = depth_first.pop()
        if 'result' in dict:
            clean(dict['result'])
            depth_first.append(dict['result'])
        if 'arg' in dict:
            clean(dict['arg'])
            depth_first.append(dict['arg'])
    return ccg

feature_re = re.compile('\[[A-Za-z]+\]')
def remove_features(tag):
    return feature_re.sub('',tag)

def get_num_args(ccg_tag):
    num_args = 0
    # adjuncts
    if 'result' in ccg_tag and remove_features(ccg_tag['result']['category'])==remove_features(ccg_tag['arg']['category']):
        return 1
    dict = ccg_tag
    while 'result' in dict:
        if remove_features(dict['result']['category']) == remove_features(dict['arg']['category']):
            num_args+=1
            break
        num_args+=1
        dict = dict['result']

    return num_args



def clean(dict):
    if 'attr' in dict:
        del dict['attr']
    if 'addr' in dict:
        del dict['addr']
    if 'addr_bin' in dict:
        del dict['addr_bin']

def main():
    amr_file = sys.argv[1]
    ccg_file = sys.argv[2]

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    # amrs = amrs[:1000]
    add_nlp_data(amrs, amr_file)

    align_file = amr_file.replace('.txt', '') + '.subgraph_alignments.json'
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file.replace('.txt', '') + '.relation_alignments.json'
    relation_alignments = reader.load_alignments_from_json(align_file, amrs)

    ccg_tags = []
    with open(ccg_file, encoding='utf8') as f:
        for line in f:
            words = line.split(' ')
            ccg = [w.split('|')[-1] for w in words]
            ccg = [get_ccg_tag(c) for c in ccg]
            ccg_tags.append(ccg)

    correct = Counter()
    total = Counter()
    confusion_matrix = {}
    for amr,ccg in zip(amrs, ccg_tags):
        for span in amr.spans:
            pos = amr.pos[span[0]]
            if len(ccg)!=len(amr.tokens):print()
            if len(span)>1: continue
            sub_align = amr.get_alignment(subgraph_alignments, token_id=span[0])
            rel_align = amr.get_alignment(relation_alignments, token_id=span[0])
            if not sub_align and not rel_align: continue

            ccg_tag = ccg[span[0]]
            if ccg_tag['category']=='-ERR-': continue
            num_args = get_num_args(ccg_tag)
            arg_edges = [(s,r,t) for s,r,t in rel_align.edges if not (s in sub_align.nodes and t in sub_align.nodes)]
            num_args2 = len(arg_edges)
            if not sub_align:
                num_args2 += 1
            if pos=='IN' and num_args!=num_args2:
                print()

            if num_args==num_args2:
                correct[pos]+=1
            total[pos]+=1
            if pos not in confusion_matrix:
                confusion_matrix[pos] = Counter()
            confusion_matrix[pos][(num_args, num_args2)]+=1

    print('total', sum(correct.values()), '/', sum(total.values()), '=', sum(correct.values()) / sum(total.values()))
    for pos in sorted(correct.keys(), reverse=True, key=lambda p:total[p]):
        print(pos, correct[pos],'/',total[pos],'=',correct[pos]/total[pos])#, f'({confusion_matrix[pos].most_common(5)})')



if __name__=='__main__':
    main()