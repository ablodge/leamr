import json
import math
import sys
from collections import Counter
from statistics import stdev, mean

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr_readers import JAMR_AMR_Reader
from tqdm import tqdm

from display import Display
from models.distance_model import Skellam_Distance_Model
from models.utils import align_all
from nlp_data import add_nlp_data
from rule_based.preprocess_subgraphs import align_fuzzy_match, align_exact_match_english, preprocess, preprocess_english

ENGLISH = True

class Subgraph_Model:

    def __init__(self, amrs, alpha=0.1):

        self.alpha = alpha
        self.translation_count = {}
        self.translation_total = 0
        self.num_sungraphs = 0

        self.tokens_count = Counter()
        self.tokens_total = 0
        for amr in amrs:
            for span in amr.spans:
                token_label = ' '.join(amr.lemmas[t] for t in span)
                self.tokens_count[token_label] += 1
        self.tokens_total = sum(self.tokens_count[t]+self.alpha for t in self.tokens_count)

        self.naive_subgraph_model = Naive_Subgraph_Model(tokens_count=self.tokens_count,
                                                         tokens_total=self.tokens_total,
                                                         alpha=alpha)
        self.naive_subgraph_model.update_parameters(amrs)

        self.distance_model = Skellam_Distance_Model()

        self._trans_logp_memo = {}


    def logp(self, amr, alignments, align):
        preprocess(amr, alignments, align)
        if ENGLISH:
            preprocess_english(amr, alignments, align)

        tokens = ' '.join(amr.lemmas[t] for t in align.tokens)
        subgraph_label = self.get_alignment_label(amr, align.nodes)
        token_logp = math.log(self.tokens_count[tokens]+self.alpha) - math.log(self.tokens_total)

        if (tokens, subgraph_label) in self._trans_logp_memo:
            trans_logp = self._trans_logp_memo[(tokens, subgraph_label)]
        elif tokens in self.translation_count and subgraph_label in self.translation_count[tokens]:
            trans_logp = math.log(self.translation_count[tokens][subgraph_label]+self.alpha) - math.log(self.translation_total)
            trans_logp -= token_logp
        else:
            trans_logp = self.naive_subgraph_model.logp(amr, align)

        if (tokens, subgraph_label) not in self._trans_logp_memo:
            self._trans_logp_memo[(tokens, subgraph_label)] = trans_logp

        dist_logp = self.distance_logp(amr, alignments, align)

        return trans_logp + dist_logp

    def get_alignment_label(self, amr, nodes):
        if not nodes:
            return 'Null'
        if len(nodes)==1:
            return amr.nodes[nodes[0]].replace(' ','_')
        edges = [(s, r, t) for s, r, t in amr.edges if s in nodes and t in nodes]
        # nodes, edges = self._remove_ignored_parts(amr, nodes, edges)
        subgraph_label = [amr.nodes[n] for n in nodes] + [f'({amr.nodes[s]},{r},{amr.nodes[t]})' for s, r, t in edges]
        subgraph_label = [s for s in sorted(subgraph_label)]
        subgraph_label = [s.replace(' ','_') for s in subgraph_label]
        subgraph_label = ' '.join(subgraph_label)
        return subgraph_label

    def distance_logp(self, amr, alignments, align):
        logp = 0
        parent_dists = []
        child_dists = []
        nodes = align.nodes
        tokens = align.tokens
        nodes = set(nodes)
        for s, r, t in amr.edges:
            if t in nodes and s not in nodes:
                salign = amr.get_alignment(alignments, node_id=s)
                dist = self.distance_model.distance(amr, salign.tokens, tokens)
                parent_dists.append(dist)
            elif s in nodes and t not in nodes:
                talign = amr.get_alignment(alignments, node_id=t)
                dist = self.distance_model.distance(amr, tokens, talign.tokens)
                t_parents = [s2 for s2,r2,t2 in amr.edges if t2==t]
                # ignore reentrancies
                if len(t_parents)>1:
                    reentrancy = False
                    for s2 in t_parents:
                        if s==s2: continue
                        s2_align = amr.get_alignment(alignments, node_id=s2)
                        other_dist = self.distance_model.distance(amr, s2_align.tokens, talign.tokens)
                        if other_dist<=dist:
                            reentrancy = True
                            break
                    if reentrancy:
                        continue
                child_dists.append(dist)
        if parent_dists:
            dist = min(parent_dists)
            logp += self.distance_model.logp(dist)
        else:
            logp += self.distance_model.logp(self.distance_model.distance_stdev)
        if child_dists:
            dist = max(child_dists)
            logp += self.distance_model.logp(dist)
        else:
            logp += self.distance_model.logp(self.distance_model.distance_stdev)
        return logp

    def get_initial_alignments(self, amrs):

        alignments = {}
        print('Preprocessing')
        for amr in tqdm(amrs, file=sys.stdout):
            alignments[amr.id] = []
            for span in amr.spans:
                alignments[amr.id].append(AMR_Alignment(type='subgraph', tokens=span, amr=amr))

            align_fuzzy_match(amr, alignments)
            if ENGLISH:
                align_exact_match_english(amr, alignments)
            for align in alignments[amr.id]:
                preprocess(amr, alignments, align)
                if ENGLISH:
                    preprocess_english(amr, alignments, align)
        return alignments

    def update_parameters(self, amrs, alignments):
        self.translation_count = {}
        self.translation_total = 0
        self._trans_logp_memo = {}

        distances = []
        subgraphs = set()

        for amr in amrs:
            if amr.id not in alignments:
                continue
            taken = set()
            for align in alignments[amr.id]:
                taken.update(align.tokens)
                tokens = ' '.join(amr.lemmas[t] for t in align.tokens)
                if tokens not in self.translation_count:
                    self.translation_count[tokens] = Counter()
                subgraph_label = self.get_alignment_label(amr, align.nodes)
                self.translation_count[tokens][subgraph_label]+=1
                subgraphs.add(subgraph_label)

            # distance stats
            for s, r, t in amr.edges:
                sa = amr.get_alignment(alignments, node_id=s)
                ta = amr.get_alignment(alignments, node_id=t)
                if sa and ta:
                    dist = self.distance_model.distance(amr, sa.tokens, ta.tokens)
                    distances.append(dist)

        self.translation_total = sum(self.translation_count[t][s] for t in self.translation_count for s in self.translation_count[t])
        self.translation_total += self.alpha*len(self.tokens_count)*len(subgraphs)
        self.num_sungraphs = len(subgraphs)

        distance_mean = mean(distances)
        distance_stdev = stdev(distances)
        self.distance_model.update_parameters(distance_mean, distance_stdev)

    def get_scores(self, amr, alignments, n, unaligned=None):

        # get candidates
        if unaligned is None:
            unaligned = self.get_unaligned(amr, alignments)
        candidate_spans = [align.tokens for align in alignments[amr.id] if not align.nodes]
        candidate_neighbors = [s for s, r, t in amr.edges if t == n and s not in unaligned] + \
                              [t for s, r, t in amr.edges if s == n and t not in unaligned]

        for n2 in candidate_neighbors[:]:
            nalign = amr.get_alignment(alignments, node_id=n2)
            if nalign.type.startswith('dupl'):
                candidate_neighbors.remove(n2)

        readable = []
        scores1 = {}
        aligns1 = {}
        for i, span in enumerate(candidate_spans):
            new_align = AMR_Alignment(type='subgraph', tokens=span, nodes=[n], amr=amr)
            replaced_align = AMR_Alignment(type='subgraph', tokens=span, nodes=[], amr=amr)
            scores1[i] = self.logp(amr, alignments, new_align) - self.logp(amr, alignments, replaced_align)
            aligns1[i] = new_align
            # readable.append(self.readable_logp(amr,alignments, new_align))
        scores2 = {}
        aligns2 = {}
        for i, neighbor in enumerate(candidate_neighbors):
            replaced_align = amr.get_alignment(alignments, node_id=neighbor)
            new_align = AMR_Alignment(type='subgraph', tokens=replaced_align.tokens, nodes=replaced_align.nodes+[n], amr=amr)
            scores2[i] = self.logp(amr, alignments, new_align) - self.logp(amr, alignments, replaced_align)
            aligns2[i] = new_align
            # readable.append(self.readable_logp(amr, alignments, new_align))

        all_scores = {}
        all_aligns = {}
        for x in scores1:
            span = tuple(aligns1[x].tokens)
            all_scores[span] = scores1[x]
            all_aligns[span] = aligns1[x]
        for x in scores2:
            span = tuple(aligns2[x].tokens)
            all_scores[span] = scores2[x]
            all_aligns[span] = aligns2[x]

        partitions = {}
        for span in all_aligns:
            partitions[span] = set(all_aligns[span].nodes)

        # readable = [r for r in sorted(readable, key=lambda x:x['score'], reverse=True)]
        return all_aligns, all_scores, partitions

    def get_unaligned(self, amr, alignments):
        unaligned = set(amr.nodes)
        for align in alignments[amr.id]:
            for n in align.nodes:
                if n in unaligned:
                    unaligned.remove(n)
        return list(unaligned)

    def readable_logp(self, amr, alignments, align):
        subgraph = self.get_alignment_label(amr, align.nodes)
        tokens = ' '.join(amr.lemmas[t] for t in align.tokens)
        logp = self.logp(amr, alignments, align)
        dist_logp = self.distance_logp(amr, alignments, align)
        tokens_logp = (math.log(self.tokens_count[tokens]+self.alpha) - math.log(self.tokens_total))
        return {'score':logp,
                'tokens':tokens,
                'subgraph':subgraph,
                'transl_logp':logp-dist_logp,
                'dist_logp':dist_logp,
                'tokens_logp':tokens_logp,
                }

class Naive_Subgraph_Model:

    def __init__(self, tokens_count, tokens_total, alpha):

        self.alpha = alpha
        self.concept_translation_count = {}
        self.concept_translation_total = 0

        self.tokens_count = tokens_count
        self.tokens_total = tokens_total

        self.concept_count = Counter()
        self.concept_total = 0

        self.tokens_rank = {t:i+1 for i,t in enumerate(sorted(self.tokens_count, key=lambda x:self.tokens_count[x], reverse=True))}

    def update_parameters(self, amrs):
        for amr in amrs:
            # concept stats
            concepts = [amr.nodes[n] for n in amr.nodes]
            for span in amr.spans:
                tokens = ' '.join(amr.lemmas[t] for t in span)
                if tokens not in self.concept_translation_count:
                    self.concept_translation_count[tokens] = Counter()
                for label in concepts:
                    self.concept_translation_count[tokens][label] += 1
            for label in concepts:
                self.concept_count[label]+=1

        self.concept_total = sum(self.concept_count[l]+self.alpha for l in self.concept_count)
        self.concept_translation_total = sum(self.concept_translation_count[t][p] for t in self.concept_translation_count for p in self.concept_translation_count[t])
        self.concept_translation_total += self.alpha*len(self.tokens_count)*len(self.concept_count)

    def logp(self, amr, align):
        AVG_SUBGRAPHS = False

        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        concepts = [amr.nodes[n] for n in align.nodes]
        if not concepts:
            return self.null_logp(token_label)
        if self.tokens_count[token_label] == 1:
            return self.rare_logp(concepts)

        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        concepts_logp = 0
        for label in concepts:
            concepts_logp += math.log(self.concept_count[label]+self.alpha) - math.log(self.concept_total)
        if AVG_SUBGRAPHS:
            concepts_logp /= len(concepts)
        joint_logp = 0
        for label in concepts:
            joint_logp += math.log(self.concept_translation_count[token_label][label]+self.alpha) - math.log(self.concept_translation_total)
            joint_logp -= token_logp
        if AVG_SUBGRAPHS:
            joint_logp /= len(concepts)
        joint_logp += token_logp
        pmi = joint_logp - token_logp - concepts_logp

        if AVG_SUBGRAPHS:
            chance_logp = - math.log(len(self.concept_count))
        else:
            chance_logp = - len(concepts) * math.log(len(self.concept_count))
        logp = self._scaled_softmax(pmi, math.exp(chance_logp))
        return logp

    def _scaled_softmax(self, value, location):
        # a function that maps (-inf,inf) to (0, 1) such that f(0) = location
        # f(x) = sigmoid(x + sigmoid^-1(location)),
        # so that f(0) = sigmoid(sigmoid^-1(location)) = location
        location = math.log(location) - math.log(1-location)
        value = value + location
        sigmoid = 1 + math.exp(-value)
        return - math.log(sigmoid)

    def rare_logp(self, concepts):
        return - len(concepts) * math.log(len(self.concept_count))

    def null_logp(self, token_label):
        if token_label in self.tokens_rank:
            rank = self.tokens_rank[token_label]
        else:
            rank = len(self.tokens_rank)
        p = 1/(math.log(rank)+1)
        return math.log(p)


def main():

    amr_file = sys.argv[1]
    cr = JAMR_AMR_Reader()
    amrs = cr.load(amr_file, remove_wiki=True)

    # amrs = amrs[:1000]

    add_nlp_data(amrs, amr_file)

    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()

    align_model = Subgraph_Model(amrs)
    iters = 10

    alignments = align_model.get_initial_alignments(amrs)

    for i in range(iters-1):
        print(f'Epoch {i}')
        alignments = align_all(align_model, amrs, alignments)
        align_model.update_parameters(amrs, alignments)

        Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.subgraphs.no-pretrain{i}.html')
    print(f'Epoch {iters-1}')
    alignments = align_all(align_model, amrs, alignments)
    align_model.update_parameters(amrs, alignments)

    Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.subgraphs.no-pretrain{iters-1}.html')

    amrs_dict = {}
    for amr in amrs:
        amrs_dict[amr.id] = amr

    for k in alignments:
        alignments[k] = [a.to_json(amrs_dict[k]) for a in alignments[k] if a.type.startswith('dupl')]
    align_file = amr_file.replace('.txt', '') + f'.subgraph_alignments.no-pretrain{iters-1}.json'
    print(f'Writing subgraph alignments to: {align_file}')
    with open(align_file, 'w+', encoding='utf8') as f:
        json.dump(alignments, f)

    # pr.disable()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats()

if __name__=='__main__':
    main()