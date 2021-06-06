import csv
import os
import sys

from amr_utils.amr_readers import AMR_Reader

from nlp_data import add_nlp_data


def load_dependencies(file, flavor='ccgbank'):

    dependencies = []
    ids = []
    with open(file, encoding='utf8') as f:
        sent = []
        for line_idx, line in enumerate(f):
            if not line.strip() or line.startswith('<\s>'):
                dependencies.append(sent)
                sent = []
                if flavor=='easysrl':
                    ids.append(len(ids))
                continue
            elif line.startswith('<s '):
                id=line.split()[1]
                id = id[len('id="'):-len('">')]
                ids.append(id)
                continue
            row = [r for r in csv.reader([line], delimiter='\t')][0]
            row = [s.strip() for s in row]
            if flavor=='ccgbank':
                words = row[4].split()
                dependency = [int(row[1]), int(row[0]), row[2], int(row[3]), words[1], words[0]]
            elif flavor=='easysrl':
                words = row[0], row[3]
                dependency = [int(row[1]), int(row[4]), row[5], int(row[6]), words[0], words[1]]
            else:
                raise Exception('Please set "flavor" to either "ccgbank" or "easysrl".')
            sent.append(dependency)
    assert len(ids) == len(dependencies)
    return ids, dependencies


def load_ccgbank(file):
    ids = []
    sents = []
    parses = []
    with open(file, encoding='utf8') as f:
        for line_idx, line in enumerate(f):
            if line.startswith('ID='):
                id = line[len('ID='):]
                id = id.split()[0]
                id = id.strip()
                ids.append(id)
                continue
            if line.strip()=='(<L NP NN NN fail N>)':
                ids.pop()
                continue
            words, tree = parse_ccgbank(line)

            for word in words:
                if word[3]:
                    word[3]['idx'] = word[0]
            reset_phrases(tree)
            add_phrases(tree)
            parses.append(tree)
            sents.append(words)
    return ids, sents, parses

def reset_phrases(tree):
    if 'word' in tree:
        return
    if 'phrase' in tree:
        del tree['phrase']
        del tree['token_ids']
    for child in tree['children']:
        reset_phrases(child)

def add_phrases(tree):
    phrase = []
    indices = []
    if 'word' in tree:
        if 'token_ids' not in tree:
            tree['token_ids'] = [tree['idx']]
        return [tree['word']], tree['token_ids']
    elif 'phrase' in tree:
        return tree['phrase'], tree['token_ids']
    else:
        for child in tree['children']:
            p, inds = add_phrases(child)
            phrase.extend(p)
            indices.extend(inds)
    if 'phrase' not in tree and 'word' not in 'tree':
        tree['phrase'] = phrase
        tree['token_ids'] = indices
    return phrase, indices

def parse_ccgbank(line):
    memory = ''
    words = []
    tree = {'supertag':None,'parent':None,'children':[],}
    current = tree
    root = True
    escaped = False
    prev = None
    idx = 0
    for ch in line:
        if ch=='(' and not escaped:
            if root:
                root = False
                continue
            else:
                current['children'].append({'supertag':None,'parent':current,'children':[],})
                current = current['children'][-1]
        elif ch==')' and (prev in ['>',' ']):
            current = current['parent']
        elif ch=='<' and not escaped:
            memory = ''
            escaped = True
        elif ch=='>' and ((memory.startswith('T') and memory.count(' ')==3) or (memory.startswith('L') and memory.count(' ')==5)):
            memory = memory.replace('   ',' . . ')
            escaped = False
            attr = memory.split()
            if attr[0]=='L':
                current['supertag'] = attr[1]
                current['word'] = attr[4]
            else:
                current['supertag'] = attr[1]
                current['head'] = int(attr[2])
            if memory.startswith('L '):
                words.append((idx, current['word'], current['supertag'], current))
                idx+=1
        else:
            memory+=ch
        prev = ch
    return words, tree


def align_ccgbank_to_sentences(ccgbank, sentences):
    new_ids = []
    new_words = []
    new_trees = []
    i = 0
    ignore = ['-RRB-', '-LRB-', '-LSB-', '-RSB-', '(', ')', '<', '>', '[', ']', '"']
    for idx, words, tree in zip(ccgbank[0], ccgbank[1], ccgbank[2]):
        # make sure sentences are aligned correctly
        # make sure sentences are aligned correctly
        test1, test2 = [w for _, w, _, _ in words if w not in ignore and '<' not in w], \
                       [w for w in sentences[i] if w not in ignore and '<' not in w]
        if test1 != test2:
            overlap = len([s for s in test1 if s in test2]) / max(len(test1), len(test2))
            difference = len([s for s in test1 if s not in test2] + [s for s in test2 if s not in test1])
            intersection = len([s for s in test1 if s in test2])
            if not (overlap >= 0.9 or (difference <= 2 and intersection >= 1)):
                count = 0
                while test1 != [w for w in sentences[i] if w not in ignore and '<' not in w]:
                    i += 1
                    count += 1
                    new_words.append([])
                    new_trees.append({})
                    new_ids.append('None')
                    if count >= 10:
                        raise Exception('Cannot find sentence for CCG parse:', idx)
        if [w[1] for w in words] != sentences[i]:
            words2 = [w if (w not in ignore and not any(c in w for c in ['(', ')', '<', '>', '[', ']'])) else '' for
                      _, w, _, _ in words]
            map = align_indices(words2, sentences[i])
            sent2 = []
            for j, tok in enumerate(sentences[i]):
                if j in map.values():
                    tag, ref = [(w[2], w[3]) for k, w in enumerate(words) if map[k] == j][0]
                else:
                    tag, ref = 'NONE', None
                sent2.append((j, tok, tag, ref))
            words = sent2
        assert len(words) == len(sentences[i])
        for word in words:
            if word[3]:
                word[3]['idx'] = word[0]
        add_phrases(tree)
        new_words.append(words)
        new_trees.append(tree)
        new_ids.append(idx)
        i += 1
    assert len(new_trees) == len(sentences)
    return new_ids, new_words, new_trees


def align_dependencies_to_sentences(dependencies, sentences):
    new_ids = []
    new_dependencies = []
    i = 0
    skip_count = 0
    for idx, deps in zip(dependencies[0], dependencies[1]):
        sent = []
        words = {}
        for dependency in deps:
            words[dependency[0]] = dependency[4]
            words[dependency[1]] = dependency[5]
            sent.append(dependency)
        ignore = ['.', ',', '?', '!', '"', '""', '""""', '(', ')',
                  'and', 'or', '@/@', '·', '[', ']', '...', '•', '-']
        # make sure sentences are aligned correctly
        while not all(j < len(sentences[i]) and words[j] == sentences[i][j] for j in words):
            sent_i = [s for s in sentences[i] if (s not in ignore or s in words.values())]
            overlap = len([words[s] for s in words if words[s] in sent_i]) / max(len(sent_i), len(words))
            difference = len([words[s] for s in words if words[s] not in sent_i] + [s for s in sent_i if
                                                                                            s not in words.values()])
            intersection = len([words[s] for s in words if words[s] in sent_i])
            if overlap >= 0.75 or (difference <= 2 and intersection >= 1):
                words2 = [words[j] if j in words else '' for j in range(max(words.keys()) + 1)]
                map = align_indices(words2, sentences[i])
                for dep in sent:
                    if map[dep[0]]!=dep[0]:
                        dep[0] = map[dep[0]]
                    if map[dep[1]]!=dep[1]:
                        dep[1] = map[dep[1]]
                # sent = []
                break
            else:
                skip_count += 1
                new_dependencies.append([])
                new_ids.append('None')
                i += 1
            if skip_count >= 10:
                raise Exception('Cannot find sentence for line:', idx)
        # new sentence
        if sent:
            skip_count = 0
        new_dependencies.append(sent)
        new_ids.append(idx)

        i += 1
    assert len(new_dependencies) == len(sentences)
    assert len(new_ids) == len(new_dependencies)
    return new_ids, new_dependencies



def align_indices(tokens1, tokens2):
    map = {}
    offset = 0
    for i,t in enumerate(tokens1):
        if tokens2[i+offset]==t or not t:
            map[i] = i+offset
        elif t in tokens2[i+offset:]:
            while tokens2[i+offset]!=t:
                offset+=1
            map[i] = i + offset
        elif any(t2.startswith(t) for t2 in tokens2[i+offset:]):
            while not tokens2[i+offset].startswith(t):
                offset+=1
            map[i] = i + offset
        else:
            raise Exception('Failed to align:\n',tokens1,'\n',tokens2)
    # pairs = [(tokens1[t],tokens2[t2]) for t,t2 in map.items()]
    return map


def load_gold_ccgs(ids_file, ccg_dependency_file, ccgbank_file):
    map_ids = {}
    with open(ids_file) as f:
        for line in f:
            if not line.strip(): continue
            line = line.split('\t')
            map_ids[line[0]] = line[1]
    ids, dependencies = load_dependencies(ccg_dependency_file)
    _, words, trees = load_ccgbank(ccgbank_file)

    new_ids = []
    new_dependencies = []
    new_words = []
    new_trees = []
    for id, deps, ws, tree in zip(ids, dependencies, words, trees):
        if id in map_ids:
            id = map_ids[id]
            new_ids.append(id)
            new_dependencies.append(deps)
            new_words.append(ws)
            new_trees.append(tree)
    return new_ids, new_dependencies, new_words, new_trees


def main():
    top_dir = sys.argv[1]
    parse_dir = sys.argv[2]
    dependency_dir = sys.argv[3]
    # os.chdir(top_dir)

    reader = AMR_Reader()
    amrs = reader.load('../data/split/train.txt', remove_wiki=True)
    amrs2 = reader.load('../data/split/dev.txt', remove_wiki=True)
    amrs3 = reader.load('../data/split/test.txt', remove_wiki=True)
    amr_ids = {'train':{' '.join(amr.tokens):amr.id for amr in amrs},
               'dev':{' '.join(amr.tokens):amr.id for amr in amrs2},
               'test':{' '.join(amr.tokens):amr.id for amr in amrs3}}
    # idx, deps = load_dependencies(r'C:\Users\Austin\OneDrive\Desktop\ccg rebank\data\PARG\00\wsj_0001.parg')
    # idx, deps = load_dependencies(r'C:\Users\Austin\OneDrive\Desktop\AMR-enhanced-alignments\data\test.ccg_dependencies.tsv', flavor='easysrl')

    ids = {}
    words = []
    for subdir_name in os.listdir(parse_dir):
        subdir = os.path.join(parse_dir, subdir_name)
        for file_name in os.listdir(subdir):
            file = os.path.join(subdir, file_name)
            idx, ccg_words, ccg_trees = load_ccgbank(file)
            for id, ws in zip(idx, ccg_words):
                ids[' '.join(w[1] for w in ws)] = id
            words.extend(ccg_words)

    with open('ids_map_train.tsv', 'w+', encoding='utf8') as f:
        for k in ['train']:
            common_sents = [(ids[i],amr_ids[k][i],i) for i in ids if i in amr_ids[k]]
            print(k, len(common_sents))
            for id1, id2, sent in common_sents:
                f.write(f'{id1}\t{id2}\t{sent}\n')
    with open('ids_map_test.tsv', 'w+', encoding='utf8') as f:
        for k in ['dev','test']:
            common_sents = [(ids[i],amr_ids[k][i],i) for i in ids if i in amr_ids[k]]
            print(k, len(common_sents))
            for id1, id2, sent in common_sents:
                f.write(f'{id1}\t{id2}\t{sent}\n')

    output_file = os.path.join(top_dir, 'ccgbank_parses.gold.txt')
    with open(output_file, 'w+', encoding='utf8') as fw:
        for subdir_name in os.listdir(parse_dir):
            subdir = os.path.join(parse_dir, subdir_name)
            for file_name in os.listdir(subdir):
                file = os.path.join(subdir, file_name)
                with open(file, 'r', encoding='utf') as fr:
                    s = fr.read()
                    fw.write(s)

    output_file = os.path.join(top_dir, 'ccgbank_dependencies.gold.txt')
    with open(output_file, 'w+', encoding='utf8') as fw:
        for subdir_name in os.listdir(dependency_dir):
            subdir = os.path.join(dependency_dir, subdir_name)
            for file_name in os.listdir(subdir):
                file = os.path.join(subdir, file_name)
                with open(file, 'r', encoding='utf') as fr:
                    s = fr.read()
                    fw.write(s)
    print()




if __name__=='__main__':
    main()