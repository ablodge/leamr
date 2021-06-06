import re
from collections import Counter, defaultdict

from amr_utils.amr_readers import AMR_Reader

from nlp_data import add_nlp_data

from evaluate.utils import table_to_latex, table_to_excel
from load_ccg import load_dependencies, load_ccgbank, align_dependencies_to_sentences, align_ccgbank_to_sentences, \
    load_gold_ccgs

VERBOSE = False
report = table_to_excel


def span_analysis(amrs, dependencies):

    span_correct = Counter()
    span_total = Counter()
    for amr, ccg_dep in zip(amrs, dependencies):
        if not ccg_dep:
            continue
        for span in amr.spans:
            if len(span) > 1:
                if is_connected(span, ccg_dep):
                    span_correct['all'] += 1
                    span_correct[len(span)] += 1
                else:
                    print()
                span_total['all'] += 1
                span_total[len(span)] += 1

    print('Table: Connected Span')
    print(report(table=[[span_total[k], 100 * span_correct[k]/span_total[k]] for k in span_total],
                 row_labels=[str(k) for k in span_total],
                 column_labels=['type', '%']))

def reentrancy_analysis(amrs, dependencies, subgraph_alignments, reentrancy_alignments):

    correct = Counter()
    total = Counter()
    for amr, ccg_dep in zip(amrs, dependencies):
        if not ccg_dep:
            continue
        reentrancies = []
        for align in reentrancy_alignments[amr.id]:
            reentrancies.extend(align.edges)
        for s,r,t in reentrancies:
            salign = amr.get_alignment(subgraph_alignments, node_id=s)
            talign = amr.get_alignment(subgraph_alignments, node_id=t)
            reent_align = amr.get_alignment(reentrancy_alignments, edge=(s,r,t))
            if any(d[0] in salign.tokens and d[1] in talign.tokens for d in ccg_dep):
                correct['all']+=1
                correct[reent_align.type]+=1
            total['all']+=1
            total[reent_align.type]+=1
    print('Table: Reentrencies')
    print(report(table=[[total[k], 100 * correct[k] / total[k]] for k, _ in total.most_common()],
                 row_labels=[k for k, _ in total.most_common()],
                 column_labels=['type', 'recall (%)']))

def scope_analysis(amrs, ccg_lex, ccg_trees, subgraph_alignments, relation_alignments, reentrancy_alignments):
    correct = Counter()
    total = 0
    for amr, ccg_words, ccg_tree in zip(amrs, ccg_lex, ccg_trees):
        if not ccg_tree: continue
        ignore = [t for t,_ in enumerate(amr.tokens) if not amr.get_alignment(subgraph_alignments, token_id=t)
                                                    and not amr.get_alignment(relation_alignments, token_id=t)]
        for span in amr.spans:
            amr_scope = amr_largest_constituent_with_head(amr, span, subgraph_alignments, relation_alignments, reentrancy_alignments)
            amr_scope = [t for t in amr_scope if t not in ignore]
            ccg_scope = ccg_largest_constituent_with_head(ccg_words[span[0]][3])
            ccg_scope = [t for t in ccg_scope if t not in ignore]
            if amr_scope == ccg_scope:
                correct['='] += 1
            elif all(t in ccg_scope for t in amr_scope):
                correct['<'] += 1
            elif all(t in amr_scope for t in ccg_scope):
                correct['>'] += 1

            total+=1
    print('Table: Scope')
    print(report(table=[[100 * correct[k] / total] for k,_ in correct.most_common()],
                 row_labels=[k for k,_ in correct.most_common()],
                 column_labels=['type', '%']))

def empty_syntax_analysis(amrs, dependencies, subgraph_alignments, relation_alignments):

    extra_syntax = Counter()
    extra_syntax_total = Counter()

    for amr, ccg_dep in zip(amrs, dependencies):
        if not ccg_dep:
            continue
        for span in amr.spans:
            sub_align = amr.get_alignment(subgraph_alignments, token_id=span[0])
            rel_align = amr.get_alignment(relation_alignments, token_id=span[0])
            rel_align.edges = [e for e in rel_align.edges
                               if not (e[0] in sub_align.nodes and e[2] in sub_align.nodes)]
            ccg_args = [d for d in ccg_dep if d[0] == span[0] and d[3]!=-1]
            amr_args = [(amr.get_alignment(subgraph_alignments, node_id=e[0]).tokens,
                         amr.get_alignment(subgraph_alignments, node_id=e[2]).tokens,
                         e) for e in rel_align.edges]
            amr_args = [(a[0], a[1], a[2], (a[0] if a[0] != span else a[1])) for a in amr_args]
            for dep in ccg_args:
                extra_syntax_total['all']+=1
                if not any((dep[0] in span and dep[1] in a[3]) for a in amr_args):
                    extra_syntax['all']+=1
    print('Table: Empty Syntax')
    print(report(table=[[100 * extra_syntax[k] / extra_syntax_total[k]] for k in extra_syntax],
                 row_labels=[k for k in extra_syntax],
                 column_labels=['type', 'recall (%)']))

def concordance_analysis(amrs, dependencies, ccg_lex, ccg_trees, subgraph_alignments, relation_alignments, reentrancy_alignments):

    correct = Counter()
    total = Counter()
    incorrect_count = Counter()
    distance_correct = Counter()
    distance_total = Counter()

    different_head_count = Counter()
    avg_dist_other = []
    prep_incorrect_details = Counter()

    for amr, ccg_dep, ccg_words, ccg_tree in zip(amrs, dependencies, ccg_lex, ccg_trees):
        if not ccg_dep:
            continue
        null_sem = [t for t, _ in enumerate(amr.tokens) if
                    not amr.get_alignment(subgraph_alignments, token_id=t)
                    and not amr.get_alignment(relation_alignments, token_id=t)]
        if VERBOSE:
            print(' '.join(amr.tokens))
        for span in amr.spans:
            lemma = ' '.join(amr.lemmas[t] for t in span)
            pos = amr.pos[span[0]]

            sub_align = amr.get_alignment(subgraph_alignments, token_id=span[0])
            rel_align = amr.get_alignment(relation_alignments, token_id=span[0])
            reentrancies = []
            for align in reentrancy_alignments[amr.id]:
                if align.type!='reentrancy:primary':
                    reentrancies.extend(align.edges)
            rel_align.edges = [e for e in rel_align.edges
                               if not (e[0] in sub_align.nodes and e[2] in sub_align.nodes)
                               and not e in reentrancies]
            if not sub_align and not rel_align:
                continue

            ccg_args = [d for d in ccg_dep if d[0] in span]
            ccg_args = [d for d in ccg_args if d[0] != d[1]]
            amr_args = [(amr.get_alignment(subgraph_alignments, node_id=e[0]).tokens,
                         amr.get_alignment(subgraph_alignments, node_id=e[2]).tokens,
                         e) for e in rel_align.edges]
            amr_args = [(a[0],a[1],a[2],(a[0] if a[0]!=span else a[1])) for a in amr_args if a[0] and a[1]]
            if not sub_align:
                a = amr_args[0]
                amr_args = [(span,a[0],a[2],a[0]), (span,a[1],a[2],a[1])]
            amr_args = [a for a in amr_args if len(set(span).intersection(a[3]))==0]
            if VERBOSE:
                print(' '.join(amr.tokens[t] for t in span),
                      ' '.join(str(ccg_words[t][2]) for t in span) if ccg_words else '')
                print('CCG Arg: ', [(amr.tokens[d[0]], d[2] + ':' + str(d[3]), amr.tokens[d[1]]) for d in ccg_args])
            for amr_arg in amr_args:
                if not lemma[0].isalpha() and not lemma[0].isdigit():
                    continue

                head = amr.get_alignment(subgraph_alignments, node_id=amr_arg[2][0]).tokens
                dep = amr.get_alignment(subgraph_alignments, node_id=amr_arg[2][2]).tokens
                dist = abs(span[0] - amr_arg[3][0])
                dist = str(dist) if dist < 5 else '5-9' if 5 <= dist <= 9 else '10+'

                if VERBOSE:
                    print('AMR Arg:',
                          '_'.join(amr.tokens[t] for t in span),
                          '(', '_'.join(amr.tokens[t] for t in head), '->', '_'.join(amr.tokens[t] for t in dep), ')',
                          '(', amr.nodes[amr_arg[2][0]], amr_arg[2][1], amr.nodes[amr_arg[2][2]], ')',
                          f'(dist={dist})')

                total['all'] += 1
                distance_total[dist] += 1
                if pos.startswith('VB'):
                    total['verb'] += 1
                elif pos.startswith('NN'):
                    total['noun'] += 1
                elif pos.startswith('JJ'):
                    total['adj'] += 1
                elif pos.startswith('RB'):
                    total['adv'] += 1
                elif pos == 'IN':
                    total['prep'] += 1
                elif pos.startswith('CC'):
                    total['conj'] += 1
                else:
                    total['other'] += 1
                if any(d[0] in span and d[1] in amr_arg[3] for d in ccg_args):
                    if VERBOSE:
                        print('correct')
                    correct['all']+=1
                    distance_correct[dist] += 1
                    if pos.startswith('VB'):
                        correct['verb'] += 1
                    elif pos.startswith('NN'):
                        correct['noun'] += 1
                    elif pos.startswith('JJ'):
                        correct['adj'] += 1
                    elif pos.startswith('RB'):
                        correct['adv'] += 1
                    elif pos == 'IN':
                        correct['prep'] += 1
                    elif pos.startswith('CC'):
                        correct['conj'] += 1
                    else:
                        correct['other'] += 1
                else:
                    label = 'other'
                    incorrect_count['all']+=1
                    head_scope = amr_largest_constituent_with_head(amr, head, subgraph_alignments, relation_alignments, reentrancy_alignments)
                    dep_scope = amr_largest_constituent_with_head(amr, dep, subgraph_alignments, relation_alignments, reentrancy_alignments)
                    other_scope = amr_largest_constituent_with_head(amr, amr_arg[3], subgraph_alignments, relation_alignments, reentrancy_alignments)
                    if amr.pos[head[0]].startswith('NN') and span == head and amr_arg[2][1] != ':domain':
                        label = 'eventive noun'
                        incorrect_count[label] += 1
                        # if len(Examples[label]) < 1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                    elif test_coordination_scope(amr, head, dep, head_scope, dep_scope, ccg_words, null_sem)[0]:
                        _, scope1, scope2 = test_coordination_scope(amr, head, dep, head_scope, dep_scope, ccg_words, null_sem)
                        label = 'coordination scope'
                        incorrect_count[label] += 1
                        if VERBOSE:
                            print(' '.join(amr.tokens), '[', ' '.join(amr.tokens[t] for t in span), ']')
                            print('coordination scope:', ' '.join(amr.tokens[t] for t in scope1), 'vs.',
                                  ' '.join(amr.tokens[t] for t in scope2))
                        # if len(Examples[label]) < 1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                    elif amr_arg[2][1] == ':domain':
                        label = 'domain'
                        incorrect_count[label] += 1
                        # if len(Examples[label])<1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                    elif any(d[0] in amr_arg[3] and d[1] in span for d in ccg_dep):
                        label = 'inverse dep'
                        incorrect_count[label] += 1
                        # if len(Examples[label]) < 1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                    elif lemma in ["n't",'not','no','none','no one','nobody','nor','neither','never']:
                        label = 'neg'
                        incorrect_count[label] += 1
                        if VERBOSE:
                            dep2 = [d for d in ccg_dep if d[0] in span]
                            dep2 = dep2[0][1] if dep2 else None
                            if dep2 is not None:
                                print(' '.join(amr.tokens), '[', ' '.join(amr.tokens[t] for t in span), ']')
                                print('neg attachment:', ' '.join(amr.tokens[t] for t in head), 'vs.',
                                      amr.tokens[dep2])
                        # if len(Examples[label]) < 1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                    elif find_different_head(other_scope, span, ccg_dep, ccg_words, null_sem)[0]:
                        _, dep2 = find_different_head(other_scope, span, ccg_dep, ccg_words, null_sem)
                        different_head_count[(amr.pos[dep[0]], amr.pos[dep2])]+=1
                        label = 'different head'
                        incorrect_count[label] += 1
                        if VERBOSE:
                            print(' '.join(amr.tokens), '[', ' '.join(amr.tokens[t] for t in span), ']')
                            print('different head:', ' '.join(amr.tokens[t] for t in other_scope),
                                  ' '.join(amr.tokens[t] for t in dep), 'vs.', amr.tokens[dep2])
                        # if len(Examples[label]) < 1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                    # elif amr.pos[dep[0]] == 'IN' and sub_align:
                    #     incorrect_count['prep: subgraph'] += 1
                    # elif not sub_align and any(d[0] in head and d[1] in dep for d in ccg_dep):
                    #     incorrect_count['prep: non-core'] += 1
                    elif amr.pos[span[0]]=='IN' or \
                            (span==head and any(amr.pos[t]=='IN' and head[0]<t<dep[0]
                                and any(d[0] == t and d[1] in dep for d in ccg_dep) for t, _ in enumerate(amr.tokens))):
                        label = 'preposition'
                        incorrect_count[label] += 1
                        # if len(Examples[label]) < 1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                        amr_is_core = (amr.pos[span[0]]!='IN' or span==head)
                        in_deps = [d for d in ccg_dep if d[1] in dep]
                        ccg_is_core = (len(in_deps)>=2 and any(amr.pos[d[0]]=='IN' for d in in_deps))
                        amr_attached = head
                        if ccg_is_core:
                            ccg_attached = [d[0] for d in in_deps if amr.pos[d[0]] != 'IN']
                            ccg_attached = ccg_attached[0] if ccg_attached else None
                        else:
                            prep = [t for t, _ in enumerate(amr.tokens)
                                    if amr.pos[t]=='IN' and (head[0]<=t<=dep[0] or t<=dep[0]<=head[0])
                                    and any(d[0] == t and d[1] in dep for d in ccg_dep)]
                            ccg_attached = None
                            if prep:
                                prep = prep[0]
                                prep_deps = [d for d in ccg_dep if d[0]==prep]
                                ccg_attached = [d[0] for d in prep_deps if d[1]<prep]
                                ccg_attached = ccg_attached[0] if ccg_attached else None

                        if amr_is_core and not ccg_is_core:
                            incorrect_count['prep: core -> non-core'] += 1
                        elif not amr_is_core and ccg_is_core:
                            incorrect_count['prep: non-core -> core'] += 1
                        elif amr.pos[span[0]]=='IN' and not any(d[0] in span and d[1] in dep for d in ccg_dep):
                            if amr_arg[3][0]<span[0]:
                                incorrect_count['prep: PP attachment'] += 1
                            else:
                                incorrect_count['prep: PP obj'] += 1
                        elif not ccg_attached:
                            incorrect_count['prep: other'] += 1
                        elif ccg_attached not in amr_attached:
                            incorrect_count['prep: PP attachment'] += 1
                        else:
                            raise Exception()

                    elif not all(t in head_scope for t in dep):
                        label = 'non-projective'
                        incorrect_count[label] += 1
                        if VERBOSE:
                            print(' '.join(amr.tokens), '[',' '.join(amr.tokens[t] for t in span),']')
                            print('non-projective:', ' '.join(amr.tokens[t] for t in head_scope), ' '.join(amr.tokens[t] for t in dep))
                        # if len(Examples[label]) < 1000:
                        #     Examples[label].append((' '.join(amr.tokens), lemma))
                    if VERBOSE:
                        print('incorrect:',label)
            if VERBOSE:
                print()
    # for k in correct:
    #     print(k, correct[k]/total[k])
    # print()
    # for k in incorrect_count:
    #     print(k, incorrect_count[k]/(total['all']-correct['all']), incorrect_count[k])
    print('Table 1: % Concordance')
    print(report(table=[[correct[k], total[k], 100*correct[k]/total[k]]  for k,_ in total.most_common()],
                         row_labels=[k for k,_ in total.most_common()],
                         column_labels=['type','count','recall (%)']))
    print('Table 2: Discordance Details')
    print(report(table=[[incorrect_count[k], 100*incorrect_count[k]/(total['all']-correct['all'])] for k,_ in incorrect_count.most_common()],
                         row_labels=[k for k,_ in incorrect_count.most_common()],
                         column_labels=['incorrect type', 'count', '%']))
    print('Table 3: Concordance by distance')
    print(report(table=[[distance_correct[k], distance_total[k], 100*distance_correct[k]/distance_total[k]] for k,_ in distance_total.most_common()],
                 row_labels=[str(k) for k,_ in distance_total.most_common()],
                 column_labels=['type', 'count', 'recall (%)']))


    print(different_head_count.most_common(25))
    # print('avg dist', sum(avg_dist_other)/len(avg_dist_other))
    # print('median dist', sorted(avg_dist_other)[len(avg_dist_other)//2])
    print()

def test_coordination_scope(amr, head, dep, head_scope, dep_scope, ccg_words, ignore):
    if amr.lemmas[head[0]] in ['and', 'or', 'but']:
        scope1 = head_scope
        scope1 = [t for t in scope1 if t not in ignore]
        scope2 = ccg_largest_constituent_with_head(ccg_words[head[0]][3])
        scope2 = [t for t in scope2 if t not in ignore]
        if scope1!=scope2:
            return True, scope1, scope2
    elif amr.lemmas[dep[0]] in ['and', 'or', 'but']:
        scope1 = dep_scope
        scope1 = [t for t in scope1 if t not in ignore]
        scope2 = ccg_largest_constituent_with_head(ccg_words[dep[0]][3])
        scope2 = [t for t in scope2 if t not in ignore]
        if scope1 != scope2:
            return True, scope1, scope2
    return False, None, None

def find_different_head(dep_scope, span, ccg_dep, ccg_words, ignore):
    dep_scope_ = [t for t in dep_scope if t not in ignore]
    for d in ccg_dep:
        if d[1] in dep_scope and d[0] in span:
            new_dep_scope_ = ccg_largest_constituent_with_head(ccg_words[d[1]][3])
            new_dep_scope_ = [t for t in new_dep_scope_ if t not in ignore]
            if dep_scope_ == new_dep_scope_:
                return True, d[1]
    return False, None

def lexical_analysis(amrs, dependencies, ccg_lex, ccg_trees, subgraph_alignments, relation_alignments, reentrancy_alignments):

    correct = Counter()
    total = Counter()

    lexical_details = Counter()
    lexical_details2 = {}

    tag_types_amr = {k:Counter() for k in ['verb','noun','prep','adj','adv']}
    tag_types_ccg = {k: Counter() for k in ['verb', 'noun', 'prep', 'adj', 'adv']}
    tag_types_ccg2 = {k: Counter() for k in ['verb', 'noun', 'prep', 'adj', 'adv']}

    for amr, ccg_dep, ccg_words, ccg_tree in zip(amrs, dependencies, ccg_lex, ccg_trees):
        if not ccg_dep:
            continue
        for span in amr.spans:
            lemma = ' '.join(amr.lemmas[t] for t in span)
            pos = amr.pos[span[0]]

            sub_align = amr.get_alignment(subgraph_alignments, token_id=span[0])
            rel_align = amr.get_alignment(relation_alignments, token_id=span[0])

            rel_align.edges = [e for e in rel_align.edges
                               if not (e[0] in sub_align.nodes and e[2] in sub_align.nodes)]

            ccg_args = [d for d in ccg_dep if d[0] in span and d[1] not in span]
            amr_args = [(amr.get_alignment(subgraph_alignments, node_id=e[0]).tokens,
                         amr.get_alignment(subgraph_alignments, node_id=e[2]).tokens,
                         e) for e in rel_align.edges]
            amr_args = [(a[0],a[1],a[2],(a[0] if a[0]!=span else a[1])) for a in amr_args if a[0] and a[1]]
            if not sub_align and rel_align:
                a = amr_args[0]
                amr_args = [(span,a[0],a[2],a[0]), (span,a[1],a[2],a[1])]
            amr_args = [a for a in amr_args if len(set(span).intersection(a[3])) == 0]

            left_amr_args = [a for a in amr_args if a[3][0]<span[0]]
            right_amr_args = [a for a in amr_args if a[3][0]>span[0]]
            left_ccg_args = [d for d in ccg_args if d[1]<d[0]]
            right_ccg_args = [d for d in ccg_args if d[1] > d[0]]
            tag_amr = 'A' + ''.join('\\B' for _ in left_amr_args) + ''.join('/C' for _ in right_amr_args)
            if not sub_align and not rel_align:
                tag_amr = 'NULL'
            tag_ccg = remove_features(ccg_words[span[0]][2])
            tag_ccg2 = 'A' + ''.join('\\B' for _ in left_ccg_args) + ''.join('/C' for _ in right_ccg_args)
            if len(span)>1:
                pos = amr.pos[span[0]]
                pos_label = 'verb' if pos.startswith('VB') else 'noun' if pos.startswith('NN') else 'prep' if pos=='IN' \
                    else 'adj' if pos.startswith('JJ') else 'adv' if pos.startswith('RB') else ''
                if pos_label:
                    tag_types_ccg[pos_label][tag_ccg]+=1
                    tag_types_amr[pos_label][tag_amr]+=1
                    tag_types_ccg2[pos_label][tag_ccg2]+=1

            if not sub_align and not rel_align:
                continue
            if not lemma[0].isalpha() and not lemma[0].isdigit():
                continue
            if VERBOSE:
                print(' '.join(amr.tokens[t] for t in span),
                      ' '.join(str(ccg_words[t][2]) for t in span) if ccg_words else '')
                print('CCG Arg: ', [(amr.tokens[d[0]], d[2] + ':' + str(d[3]), amr.tokens[d[1]]) for d in ccg_args])

            if len(left_ccg_args+right_ccg_args)>len(left_amr_args+right_amr_args):
                correct['<']+=1
            elif len(left_ccg_args+right_ccg_args)<len(left_amr_args+right_amr_args):
                correct['>'] += 1
            if len(left_amr_args)==len(left_ccg_args) and len(right_amr_args)==len(right_ccg_args):
                correct['all']+=1
                if pos.startswith('VB'):
                    correct['verb'] += 1
                elif pos.startswith('NN'):
                    correct['noun'] += 1
                elif pos.startswith('JJ'):
                    correct['adj'] += 1
                elif pos.startswith('RB'):
                    correct['adv'] += 1
                elif pos == 'IN':
                    correct['prep'] += 1
                elif pos.startswith('CC'):
                    correct['conj'] += 1
                else:
                    correct['other'] += 1
            else:
                if tag_ccg not in lexical_details2:
                    lexical_details2[tag_ccg] = Counter()
                lexical_details2[tag_ccg][tag_amr]+=1
                lexical_details[tag_ccg]+=1
            total['all'] += 1
            total['>']+=1
            total['<']+=1
            if pos.startswith('VB'):
                total['verb'] += 1
            elif pos.startswith('NN'):
                total['noun'] += 1
            elif pos.startswith('JJ'):
                total['adj'] += 1
            elif pos.startswith('RB'):
                total['adv'] += 1
            elif pos == 'IN':
                total['prep'] += 1
            elif pos.startswith('CC'):
                total['conj'] += 1
            else:
                total['other'] += 1
            if VERBOSE:
                print()
    # for k in correct:
    #     print(k, correct[k]/total[k])
    # print()
    # for k in incorrect_count:
    #     print(k, incorrect_count[k]/(total['all']-correct['all']), incorrect_count[k])
    print('Table 1: Lexical Analysis')
    print(report(table=[[correct[k], total[k], 100*correct[k]/total[k]]  for k,_ in total.most_common()],
                         row_labels=[k for  k,_ in total.most_common()],
                         column_labels=['type','%']))
    print('Table 2: Incorrect Details')
    print(report(table=[[lexical_details[k], 100 * lexical_details[k] / (total['all']-correct['all']),
                         [l for l in lexical_details2[k].most_common(3)]] for k,_ in lexical_details.most_common(10)],
                 row_labels=[k for k,_ in lexical_details.most_common(10)],
                 column_labels=['type', '%', 'ex']))
    # for k in tag_types_amr:
    #     total = sum(tag_types_amr[k].values())
    #     total2 = sum(tag_types_ccg2[k].values())
    #     print(k, [(k2, 100*v/total, 100*tag_types_ccg2[k][k2]/total2) for k2,v in tag_types_amr[k].most_common(5)])
    # print()
    # for k in tag_types_ccg:
    #     print(k, [k2 for k2, _ in tag_types_ccg[k].most_common(20)])
    # # print(lexical_details.most_common(25))
    # print()


def get_head(phrase):
    while 'word' not in phrase:
        phrase = phrase['children'][phrase['head']]
    return phrase

def preprocess_dependencies(deps, words, lemmas, pos):
    preprocess_dependencies_(deps, words, lemmas, pos)
    for word in words:
        idx, tok, tag, ref = word
        # coordination
        if tag=='conj':
            coord_phrase = ref
            # print(lemmas[idx])
            complete = False
            coord_tag = None
            coord_args = []
            expected_args = 0
            while True:
                coord_phrase = coord_phrase['parent']
                if len(coord_phrase['children'])<2:
                    if coord_phrase['parent'] is None:
                        break
                    continue
                child1 = coord_phrase['children'][0]
                child2 = coord_phrase['children'][1]
                conj_idx = 0
                if idx not in child1['token_ids']:
                    child1, child2 = child2, child1
                    conj_idx = 1

                if child1['supertag'] == 'conj' and child2['supertag'] not in ['conj/conj', 'conj\\conj']:
                    coord_args.append(get_head(coord_phrase)['idx'])
                    coord_phrase['head'] = conj_idx
                    coord_tag = remove_features(child2['supertag'])
                    expected_args = remove_features(coord_phrase['supertag']).count(coord_tag)
                    arg = get_head(child2)
                    deps.append([idx, arg['idx'], 'conj', expected_args, tok, arg['word']])
                elif remove_features(child2['supertag']) == coord_tag:
                    coord_args.append(get_head(coord_phrase)['idx'])
                    coord_phrase['head'] = conj_idx
                    arg = get_head(child2)
                    deps.append([idx, arg['idx'], 'conj', expected_args-len(coord_args)+1, tok, arg['word']])
                    if expected_args == len(coord_args):
                        complete = True
                if complete or coord_phrase['parent'] is None:
                    break
            # print(coord_phrase['phrase'])
            for arg in coord_args:
                for d in deps[:]:
                    # args
                    if d[0] == arg and d[1] not in coord_phrase['token_ids']:
                        for arg2 in coord_args:
                            if not any(d2[0]==arg2 and d2[1]==d[1] for d2 in deps):
                                deps.append([arg2, d[1], d[2], d[3], words[arg2][1], d[5]])
                    # preds, modifiers
                    elif d[1] == arg and d[0] not in coord_phrase['token_ids']:
                        deps.append([d[0], idx, d[2], d[3], d[4], tok])
                        ignore_deps = [d2 for d2 in deps if d2[0]==d[0] and d2[1] in coord_args]
                        for dep in ignore_deps:
                            if dep in deps:
                                deps.remove(dep)
            # for arg in [idx]+coord_args:
            #     print([d for d in deps if d[0]==arg or d[1]==arg])
    preprocess_dependencies_(deps, words, lemmas, pos)
    # remove extra edges from adverbs
    for dep in deps[:]:
        if dep[0]==dep[1]:
            deps.remove(dep)
        elif pos[dep[0]] in ['RB','RBR','RBS','MD'] and len([d for d in deps if d[0]==dep[0]])>1 and dep[3]==1:
            deps.remove(dep)
        elif pos[dep[0]]=='IN' and len([d for d in deps if d[0]==dep[0]])>=3 and dep[3]==1 and '\\NP' in dep[2]:
            deps.remove(dep)

feat_re = re.compile('\[[a-zA-Z0-9]+\]')
def remove_features(tag):
    return feat_re.sub('', tag)

def preprocess_dependencies_(deps, words, lemmas, pos):
    for d in sorted(deps, key=lambda x:(x[0],x[1]), reverse=True):
        for d2 in deps:
            if d==d2: continue
            if d[1]==d2[0] and not any(d3[0]==d[0] and d3[1]==d2[1] for d3 in deps):
                head = d[0]
                old_dep = d[1]
                new_dep = d2[1]
                # copula heads
                if lemmas[old_dep]=='be' and any(pos[new_dep].startswith(pre) for pre in ['VB','JJ','IN','NN']) and old_dep<new_dep:
                    # print(lemmas[head], lemmas[old_dep], lemmas[new_dep], f'({" ".join(w[1] for w in words)})')
                    move_dependent(deps, head, old_dep, new_dep, words)
                # preposition heads
                elif pos[old_dep]=='IN' and any(pos[head].startswith(pre) for pre in ['VB','NN','JJ']) and old_dep<new_dep:
                    # print(lemmas[head], lemmas[old_dep], lemmas[new_dep], f'({" ".join(w[1] for w in words)})')
                    move_dependent(deps, head, old_dep, new_dep, words)
                # determiner heads
                elif pos[old_dep] == 'DT' and old_dep<new_dep:
                    # print(lemmas[head], lemmas[old_dep], lemmas[new_dep], f'({" ".join(w[1] for w in words)})')
                    move_dependent(deps, head, old_dep, new_dep, words)
            elif d[1]==d2[1] and d[0]<d2[0] and not any(d3[0]==d[0] and d3[1]==d2[0] for d3 in deps):
                head = d[0]
                old_dep = d[1]
                new_dep = d2[0]
                # modal
                if pos[old_dep].startswith('VB') and pos[new_dep] == 'MD' and new_dep<old_dep and lemmas[new_dep] not in ['will','would']:
                    # print(lemmas[head], lemmas[old_dep], lemmas[new_dep], f'({" ".join(w[1] for w in words)})')
                    move_dependent(deps, head, old_dep, new_dep, words)


def move_dependent(deps, head, old_dep, new_dep, words):
    for i,d in enumerate(deps[:]):
        if d[0]==head and d[1]==old_dep and not any(d2[0]==head and d2[1]==new_dep for d2 in deps):
            deps[i] = [head, new_dep, d[2], d[3], words[head][1], words[new_dep][1]]
            return True
    return False


def is_connected(span, dependencies):
    clusters = {t:{t} for t in span}
    for dep in dependencies:
        if dep[0] in span and dep[1] in span:
            clusters[dep[0]].update(clusters[dep[1]])
            clusters[dep[1]].update(clusters[dep[0]])
    if all(len(c)==len(span) for c in clusters.values()):
        return True
    return False

def ccg_largest_constituent_with_head(tree):
    current = tree
    scope = []
    while True:
        if current is None or get_head(current)['idx']!=tree['idx']:
            break
        scope = current['token_ids'] if 'token_ids' in current else [tree['idx']]
        current = current['parent']
    return scope



def amr_largest_constituent_with_head(amr, head, subgraph_alignments, rel_alignments, reent_alignments):
    align = amr.get_alignment(subgraph_alignments, token_id=head[0])
    if not align:
        align = amr.get_alignment(rel_alignments, token_id=head[0])
        if not align: return []
    scope = set(align.tokens)

    reentrancies = []
    for a in reent_alignments[amr.id]:
        if a.type != 'reentrancy:primary':
            reentrancies.extend(a.edges)
    desc = set()
    nodes = align.nodes
    if not nodes:
        nodes = align.edges[0][2]
    desc.update(nodes)
    root = min(nodes)
    while True:
        found = False
        for s,r,t in amr.edges:
            if (s,r,t) in reentrancies: continue
            if s in desc:
                # if t==root:
                #     raise Exception('Found Cycle')
                if t not in desc:
                    found = True
                    desc.add(t)
        if not found:
            break
    # check on the left
    index = [i for i,span in enumerate(amr.spans) if align.tokens[0] in span][0]
    left_spans = amr.spans[:index]
    left_spans.reverse()
    for span in left_spans:
        left_align = amr.get_alignment(subgraph_alignments, token_id=span[0])
        if not left_align or all(n in desc for n in left_align.nodes):
            scope.update(span)
        else:
            break
    # check on the left
    right_spans = amr.spans[index+1:]
    for span in right_spans:
        right_align = amr.get_alignment(subgraph_alignments, token_id=span[0])
        if not right_align or all(n in desc for n in right_align.nodes):
            scope.update(span)
        else:
            break

    scope = [i for i in sorted(scope)]
    return scope

def load_data1():
    amr_file = '../data/split/train.txt'
    ccg_dependency_file = '../data/train.ccg_dependencies.tsv'
    ccgbank_file = '../data/train.ccg_parse.txt'

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    add_nlp_data(amrs, amr_file)

    # predicted data
    align_file = amr_file.replace('.txt', '') + '.subgraph_alignments.json'
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file.replace('.txt', '') + '.relation_alignments.json'
    relation_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file.replace('.txt', '') + '.reentrancy_alignments.json'
    reentrancy_alignments = reader.load_alignments_from_json(align_file, amrs)

    sentences = [amr.tokens for amr in amrs]
    _, dependencies = align_dependencies_to_sentences(load_dependencies(ccg_dependency_file, flavor='easysrl'), sentences)
    _, ccg_lex, ccg_trees = align_ccgbank_to_sentences(load_ccgbank(ccgbank_file), sentences)

    return amrs, subgraph_alignments, relation_alignments, reentrancy_alignments, dependencies, ccg_lex, ccg_trees

def load_data2():
    amr_file1 = '../data/split/dev.txt'
    amr_file2 = '../data/split/test.txt'
    ccg_dependency_file = '../data/test.ccg_dependencies.tsv'
    ccgbank_file = '../data/test.ccg_parse.txt'

    reader = AMR_Reader()
    amrs = reader.load(amr_file1, remove_wiki=True)
    add_nlp_data(amrs, amr_file1)
    amrs2 = reader.load(amr_file2, remove_wiki=True)
    add_nlp_data(amrs2, amr_file2)
    amrs+=amrs2

    # gold data
    align_file = amr_file1.replace('.txt', '') + '.subgraph_alignments.gold.json'
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file1.replace('.txt', '') + '.relation_alignments.gold.json'
    relation_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file1.replace('.txt', '') + '.reentrancy_alignments.gold.json'
    reentrancy_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file2.replace('.txt', '') + '.subgraph_alignments.gold.json'
    subgraph_alignments.update(reader.load_alignments_from_json(align_file, amrs))
    align_file = amr_file2.replace('.txt', '') + '.relation_alignments.gold.json'
    relation_alignments.update(reader.load_alignments_from_json(align_file, amrs))
    align_file = amr_file2.replace('.txt', '') + '.reentrancy_alignments.gold.json'
    reentrancy_alignments.update(reader.load_alignments_from_json(align_file, amrs))

    sentences = [amr.tokens for amr in amrs]
    _, dependencies = align_dependencies_to_sentences(load_dependencies(ccg_dependency_file, flavor='easysrl'), sentences)
    _, ccg_lex, ccg_trees = align_ccgbank_to_sentences(load_ccgbank(ccgbank_file), sentences)

    return amrs, subgraph_alignments, relation_alignments, reentrancy_alignments, dependencies, ccg_lex, ccg_trees


def load_data3():
    amr_file = '../data/split/train.txt'
    ccg_dependency_file = '../data/ccg/ccgbank_dependencies.gold.txt'
    ccgbank_file = '../data/ccg/ccgbank_parses.gold.txt'
    ids_file = '../data/ccg/ids_map_train.tsv'

    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)
    add_nlp_data(amrs, amr_file)

    # predicted data
    align_file = amr_file.replace('.txt', '') + '.subgraph_alignments.json'
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file.replace('.txt', '') + '.relation_alignments.json'
    relation_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file.replace('.txt', '') + '.reentrancy_alignments.json'
    reentrancy_alignments = reader.load_alignments_from_json(align_file, amrs)

    ids, dependencies, ccg_lex, ccg_trees = load_gold_ccgs(ids_file, ccg_dependency_file, ccgbank_file)
    amrs = {amr.id: amr for amr in amrs}
    amrs2 = []
    for id in ids:
        amrs2.append(amrs[id])
    amrs = amrs2
    return amrs, subgraph_alignments, relation_alignments, reentrancy_alignments, dependencies, ccg_lex, ccg_trees


def load_data4():
    amr_file1 = '../data/split/dev.txt'
    amr_file2 = '../data/split/test.txt'
    ccg_dependency_file = '../data/ccg/ccgbank_dependencies.gold.txt'
    ccgbank_file = '../data/ccg/ccgbank_parses.gold.txt'
    ids_file = '../data/ccg/ids_map_test.tsv'

    reader = AMR_Reader()
    amrs = reader.load(amr_file1, remove_wiki=True)
    add_nlp_data(amrs, amr_file1)
    amrs2 = reader.load(amr_file2, remove_wiki=True)
    add_nlp_data(amrs2, amr_file2)
    amrs += amrs2

    # gold data
    align_file = amr_file1.replace('.txt', '') + '.subgraph_alignments.gold.json'
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file1.replace('.txt', '') + '.relation_alignments.gold.json'
    relation_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file1.replace('.txt', '') + '.reentrancy_alignments.gold.json'
    reentrancy_alignments = reader.load_alignments_from_json(align_file, amrs)
    align_file = amr_file2.replace('.txt', '') + '.subgraph_alignments.gold.json'
    subgraph_alignments.update(reader.load_alignments_from_json(align_file, amrs))
    align_file = amr_file2.replace('.txt', '') + '.relation_alignments.gold.json'
    relation_alignments.update(reader.load_alignments_from_json(align_file, amrs))
    align_file = amr_file2.replace('.txt', '') + '.reentrancy_alignments.gold.json'
    reentrancy_alignments.update(reader.load_alignments_from_json(align_file, amrs))

    ids, dependencies, ccg_lex, ccg_trees = load_gold_ccgs(ids_file, ccg_dependency_file, ccgbank_file)
    amrs = {amr.id: amr for amr in amrs}
    amrs2 = []
    for id in ids:
        amrs2.append(amrs[id])
    amrs = amrs2
    return amrs, subgraph_alignments, relation_alignments, reentrancy_alignments, dependencies, ccg_lex, ccg_trees


def main():
    amrs, subgraph_alignments, relation_alignments, reentrancy_alignments, dependencies, ccg_lex, ccg_trees = load_data1()

    for amr, ccg_dep, ccg_words, ccg_tree in zip(amrs, dependencies, ccg_lex, ccg_trees):
        if not ccg_dep:
            continue
        preprocess_dependencies(ccg_dep, ccg_words, amr.lemmas, amr.pos)
        # preprocess_dependencies(ccg_dep, ccg_words, amr.lemmas, amr.pos)

    blank_sents1 = len([_ for _ in ccg_trees if not _])
    blank_sents2 = len([_ for _ in dependencies if not _])
    print('Total Sentences: ',len(amrs))
    print('Sentences with Parses: ', len(amrs)-blank_sents1)
    print('Sentences with Dependencies: ', len(amrs) - blank_sents2)

    # TODO
    # 1. Do Lexical Arg Structs Match (single token)
    # 2. Do Lexical Arg Structs Match (multiple token)
    # 2. Do Lexical Arg Structs Match (single relations)
    # 3. Do Dependencies Match (no reentrancies)
    # 4. Do reentrancies Match (control)
    # 5. Do reentrancies Match (raising)
    # 6. Do reentrancies Match (coref)
    # concordance_analysis(amrs, dependencies, ccg_lex, ccg_trees, subgraph_alignments, relation_alignments, reentrancy_alignments)
    span_analysis(amrs, dependencies)
    # reentrancy_analysis(amrs, dependencies, subgraph_alignments, reentrancy_alignments)
    # empty_syntax_analysis(amrs, dependencies, subgraph_alignments, relation_alignments)
    # lexical_analysis(amrs, dependencies, ccg_lex, ccg_trees, subgraph_alignments, relation_alignments, reentrancy_alignments)
    # scope_analysis(amrs, ccg_lex, ccg_trees, subgraph_alignments, relation_alignments, reentrancy_alignments)
    print()

if __name__=='__main__':
    main()