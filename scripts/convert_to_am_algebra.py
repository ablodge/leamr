import csv
import sys

from amr_utils.alignments import convert_alignment_to_subgraph
from amr_utils.amr_readers import AMR_Reader

from nlp_data import add_nlp_data
from rule_based.relation_rules import rule_based_align_all_relations


def load_conll(file):
    conll = []
    with open(file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        sent = []
        for row in spamreader:
            if not row:
                conll.append(sent)
                sent = []
            else:
                sent.append(row)
    return conll

def write_conll(file, conll):
    with open(file, 'w+', encoding='utf8') as f:
        for sent in conll:
            for row in sent:
                f.write('\t'.join(row)+'\n')
            f.write('\n')


def get_replacement():
    return '_'

def special_graph_string(amr, node_ids):
    amr_string = f'[[{amr.root}]]'

    depth = 1
    nodes = {amr.root}
    completed = set()
    while '[[' in amr_string:
        for n in nodes.copy():
            id = node_ids[n]
            concept = amr.nodes[n] if n in node_ids and amr.nodes[n] else 'None'
            edges = sorted([e for e in amr.edges if e[0] == n], key=lambda x: x[1])
            targets = set(t for s, r, t in edges)
            edges = [f'{r} [[{t}]]' for s, r, t in edges]
            children = f' '.join(edges)
            if children:
                children = f' ' + children
            if n not in completed:
                if (concept[0].isalpha() and concept not in ['imperative', 'expressive', 'interrogative']) or targets:
                    amr_string = amr_string.replace(f'[[{n}]]', f'({id} / {concept}{children})', 1)

                else:
                    if concept == '<var>' and not children:
                        amr_string = amr_string.replace(f'[[{n}]]', f'({id})', 1)
                    else:
                        amr_string = amr_string.replace(f'[[{n}]]', f'{concept}')
                completed.add(n)
            amr_string = amr_string.replace(f'[[{n}]]', f'{id}')
            nodes.remove(n)
            nodes.update(targets)
        depth += 1
    if len(completed) < len(amr.nodes):
        missing_nodes = [n for n in amr.nodes if n not in completed]
        missing_edges = [(s, r, t) for s, r, t in amr.edges if s in missing_nodes or t in missing_nodes]
        missing_nodes= ', '.join(f'{n} / {amr.nodes[n]}' for n in missing_nodes)
        missing_edges = ', '.join(f'{s} / {amr.nodes[s]} {r} {t} / {amr.nodes[t]}' for s,r,t in missing_edges)
        print('[amr]', 'Failed to print AMR, '
              + str(len(completed)) + ' of ' + str(len(amr.nodes)) + ' nodes printed:\n '
              + str(amr.id) +':\n'
              + amr_string + '\n'
              + 'Missing nodes: ' + missing_nodes +'\n'
              + 'Missing edges: ' + missing_edges +'\n',
              file=sys.stderr)
    if not amr_string.startswith('('):
        amr_string = '(' + amr_string + ')'
    if len(amr.nodes) == 0:
        amr_string = '()'
    return amr_string


def get_supertag(amr, align, lemma):
    if len(align.nodes)==0:
        return '_'
    sub_amr = convert_alignment_to_subgraph(align, amr)
    root = [n for n in sorted(align.nodes)][0]
    for s,r,t in sub_amr.edges[:]:
        if t == root:
            sub_amr.edges.remove((s,r,t))
            s,r,t = t, (r+'-of').replace('-of-of',''), s
            sub_amr.edges.append((s, r, t))
            sub_amr.root = root
    node_ids = {}
    taken = set()
    for n in amr.nodes:
        letter = amr.nodes[n][0] if amr.nodes[n][0].isalpha() else 'x'
        if letter in node_ids:
            i = 2
            while f'{letter}{i}' in taken:
                i+=1
            idx = f'{letter}{i}'
            taken.add(idx)
            node_ids[n] = idx
        else:
            idx = f'{letter}'
            taken.add(idx)
            node_ids[n] = idx
    node_ids[root] += '<root>'
    for s,r,t in align.edges:
        node = None
        if sub_amr.nodes[s] == '<var>':
            node = s
        elif sub_amr.nodes[t] == '<var>':
            node = t
        if node:
            label = get_arg_type(align, (s,r,t))
            node_ids[node] += f'<{label}>'
    node, lex = get_lex(amr, align, lemma)
    if node:
        sub_amr.nodes[node] = 'LEX'

    supertag = special_graph_string(sub_amr, node_ids)
    supertag = supertag.replace('LEX','--LEX--')
    return supertag

def get_lex(amr, align, lemma):
    if not align.nodes:
        return None, '_'
    token_label = lemma
    for n in align.nodes:
        if amr.nodes[n].startswith(token_label):
            return n, '$LEMMA$' + amr.nodes[n].replace(token_label, '')
    root = [n for n in sorted(align.nodes)][0]
    return root, amr.nodes[root]


def get_asgraph(align):
    if not align.nodes:
        return '_'
    if not align.edges:
        return '()'
    edge_labels = []
    for s,r,t in align.edges:
        if s in align.nodes and t in align.nodes:
            continue
        label = get_arg_type(align, (s,r,t))
        edge_labels.append(label)
    edge_labels = [f'{e}()' for e in edge_labels]
    return f'({", ".join(e for e in edge_labels)})'


def get_head(amr, alignments, align):
    if not align.nodes:
        return '0'
    root = [n for n in sorted(align.nodes)][0]
    if root == amr.root:
        return '0'
    parent = [(s, r, t) for s, r, t in amr.edges if root == t][0]
    s,r,t = parent
    salign = amr.get_alignment(alignments, node_id=s)
    if salign:
        idx = amr.spans.index(salign.tokens)
        return f'{idx+1}'
    return '0'


def get_deprel(amr, alignments, align):
    if not align.nodes:
        return 'IGNORE'
    root = [n for n in sorted(align.nodes)][0]
    if root == amr.root:
        return 'ROOT'
    parent = [(s,r,t) for s,r,t in amr.edges if root == t][0]
    palign = amr.get_alignment(alignments, edge=parent)
    label = get_arg_type(palign, parent)
    if parent in align.edges:
        return f'MOD_{label}'
    else:
        return f'APP_{label}'

def get_arg_type(align, edge):
    # allowed = ['opX','sntX','s','o','oX','mod','poss']
    s,r,t = edge
    if s in align.nodes and t in align.nodes:
        return None

    if t in align.nodes and not any(r.startswith(prefix) for prefix in [':ARG',':op',':snt']):
        return 'mod'
    elif r == ':poss':
        return 'poss'
    elif r.startswith(':op') or r.startswith(':snt'):
        return r.replace(':','')
    else:
        ordered_edges = [(s,r,t) for s,r,t in align.edges if t not in align.nodes
                         or any(r.startswith(prefix) for prefix in [':ARG',':op',':snt'])]
        ordered_edges = [(s,r,t) for s,r,t in sorted(ordered_edges, key=lambda x:x[1])]
        idx = ordered_edges.index(edge)
        labels = ['s','o'] + [f'o{i}' for i in range(2,20)]
        return labels[idx]



ID = 0
FORM = 1
REPLACEMENT = 2 # _name_, _number_, _date_ etc.
LEMMA = 3
# POS = 4
# NER = 5
SUPERTAG = 6 #
LEX = 7
ASGRAPH = 8
HEAD = 9
DEPREL = 10
ALIGNED = 11

def main():
    conll_file = sys.argv[1]
    conll = load_conll(conll_file)

    amr_file = sys.argv[2]
    reader = AMR_Reader()
    amrs = reader.load(amr_file, remove_wiki=True)

    add_nlp_data(amrs, amr_file)

    align_file = sys.argv[3]
    subgraph_alignments = reader.load_alignments_from_json(align_file, amrs)

    for amr in amrs:
        rule_based_align_all_relations(amr, subgraph_alignments)
    amrs = {amr.id:amr for amr in amrs}

    new_conll = []
    all_deprels = set()
    for sent in conll:
        amr_id = [row[0] for row in sent if len(row)==1 and row[0].startswith('#id')]
        amr_id = amr_id[0].replace('#id:','')
        amr = amrs[amr_id]
        span_id = 0
        conll_tokens = [row[FORM] for row in sent if len(row)>1]
        my_tokens = ['_'.join(amr.tokens[t] for t in span) for span in amr.spans]
        if conll_tokens != my_tokens:
            continue
        sent = sent.copy()
        for row in sent:
            if len(row)==1: continue
            align = amr.get_alignment(subgraph_alignments, token_id=amr.spans[span_id][0])
            node, lex = get_lex(amr, align, row[LEMMA])
            supertag = get_supertag(amr, align, row[LEMMA])
            asgraph = get_asgraph(align)
            replacement = get_replacement()
            head = get_head(amr, subgraph_alignments, align)
            deprel = get_deprel(amr, subgraph_alignments, align)
            old_row = row.copy()
            row[LEX] = lex
            row[SUPERTAG] = supertag
            row[ASGRAPH] = asgraph
            row[REPLACEMENT] = replacement
            row[HEAD] = head
            row[DEPREL] = deprel
            new_row = row.copy()
            # new_row[SUPERTAG] = old_row[SUPERTAG]
            # if new_row!=old_row:
            #     print(old_row)
            #     print(new_row)
            #     print()
            all_deprels.add(old_row[DEPREL])
            span_id += 1
        new_conll.append(sent)
    write_conll(conll_file.replace('.amconll','2.amconll'), new_conll)

if __name__=='__main__':
    main()