import json
import math
import sys
from collections import Counter
from statistics import stdev, mean

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr_readers import JAMR_AMR_Reader

from display import Display
from models.distance_model import Skellam_Distance_Model
from models.utils import align_all
from nlp_data import add_nlp_data
from rule_based.subgraph_rules import subgraph_fuzzy_align, subgraph_exact_align_english, postprocess_subgraph, \
    postprocess_subgraph_english, clean_subgraph

ENGLISH = True

class Subgraph_Model:

    def __init__(self, amrs, alpha=1, ignore_duplicates=True):

        self.alpha = alpha
        self.ignore_duplicates = ignore_duplicates
        self.translation_count = {}
        self.translation_total = 0
        self.num_sungraphs = 1

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


    def base_logp(self, amr, alignments, align):
        postprocess_subgraph(amr, alignments, align)
        if ENGLISH:
            postprocess_subgraph_english(amr, alignments, align)
        align = clean_subgraph(amr, alignments, align)
        if align is None: return float('-inf')

        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        subgraph_label = self.get_alignment_label(amr, align.nodes)
        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)

        if (token_label, subgraph_label) in self._trans_logp_memo:
            trans_logp = self._trans_logp_memo[(token_label, subgraph_label)]
        elif token_label in self.translation_count and subgraph_label in self.translation_count[token_label]:
            trans_logp = math.log(self.translation_count[token_label][subgraph_label]+self.alpha) - math.log(self.translation_total)
            trans_logp -= token_logp
        elif len(token_label.split())>1 and any(t in self.translation_count and subgraph_label in self.translation_count[t] for t in token_label.split()):
            max_logp = float('-inf')
            for tok in token_label.split():
                if tok not in self.translation_count: continue
                logp = math.log(self.translation_count[tok][subgraph_label] + self.alpha) - math.log(self.translation_total)
                logp -= token_logp
                if logp > max_logp:
                    max_logp = logp
            trans_logp = max_logp
        else:
            trans_logp = self.naive_subgraph_model.logp(amr, align)

        if (token_label, subgraph_label) not in self._trans_logp_memo:
            self._trans_logp_memo[(token_label, subgraph_label)] = trans_logp

        return trans_logp

    def logp(self, amr, alignments, align):
        logp_subgraph_for_tokens = self.base_logp(amr, alignments, align)
        dist_logp = self.distance_logp(amr, alignments, align)
        return logp_subgraph_for_tokens + dist_logp

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
        for j, amr in enumerate(amrs):
            print(f'\r{j} / {len(amrs)} preprocessed', end='')
            alignments[amr.id] = []
            for span in amr.spans:
                alignments[amr.id].append(AMR_Alignment(type='subgraph', tokens=span, amr=amr))

            # subgraph_fuzzy_align(amr, alignments)
            # if ENGLISH:
            #     subgraph_exact_align_english(amr, alignments)
            # for align in alignments[amr.id]:
            #     postprocess_subgraph(amr, alignments, align)
            #     if ENGLISH:
            #         postprocess_subgraph_english(amr, alignments, align)
            #     test = clean_subgraph(amr, alignments, align)
            #     if test is None:
            #         align.nodes.clear()
        print('\r', end='')
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
        self.num_subgraphs = len(subgraphs)

        distance_mean = mean(distances)
        distance_stdev = stdev(distances)
        self.distance_model.update_parameters(distance_mean, distance_stdev)

    def align(self, amr, alignments, n, unaligned=None):

        # get candidates
        if unaligned is None:
            unaligned = self.get_unaligned(amr, alignments)
        candidate_spans = [align.tokens for align in alignments[amr.id] if not align.nodes]
        tmp_align = AMR_Alignment(type='subgraph', tokens=[0], nodes=[n])
        postprocess_subgraph(amr, alignments, tmp_align)
        candidate_neighbors = [s for s, r, t in amr.edges if t in tmp_align.nodes and s not in unaligned] + \
                              [t for s, r, t in amr.edges if s in tmp_align.nodes and t not in unaligned]
        candidate_neighbors = [n2 for n2 in candidate_neighbors if amr.get_alignment(alignments, node_id=n2)]

        # handle "never => ever, -" and other similar cases
        edge_map = {n:[] for n in amr.nodes}
        for s,r,t in amr.edges:
            edge_map[s].append(t)
        if not edge_map[n]:
            for n2 in amr.nodes:
                if edge_map[n2]: continue
                if n2 in unaligned: continue
                if amr.nodes[n] == amr.nodes[n2]: continue
                nalign = amr.get_alignment(alignments, node_id=n2)
                if len(nalign.nodes)!=1: continue
                if any(n in edge_map[p] and n2 in edge_map[p] for p in amr.nodes):
                    candidate_neighbors.append(n2)


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

        if not all_scores:
            return None, None

        best_span = max(all_scores.keys(), key=lambda x:all_scores[x])
        best_score = all_scores[best_span]
        best_align = all_aligns[best_span]

        # readable = [r for r in sorted(readable, key=lambda x:x['score'], reverse=True)]
        return best_align, best_score


    def get_unaligned(self, amr, alignments):
        unaligned = set(amr.nodes)
        for align in alignments[amr.id]:
            for n in align.nodes:
                if n in unaligned:
                    unaligned.remove(n)
        if self.ignore_duplicates:
            duplicates = [n for n in unaligned if len([n2 for n2 in amr.nodes if amr.nodes[n2]==amr.nodes[n]])>1]
            for n in duplicates:
                unaligned.remove(n)
        return list(unaligned)

    def readable_logp(self, amr, alignments, align):
        subgraph = self.get_alignment_label(amr, align.nodes)
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        tokens_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        subgraph_logp = self.naive_subgraph_model.subgraph_prior(amr, align)

        logp1 = self.base_logp(amr, alignments, align)
        logp2 = logp1 + tokens_logp - subgraph_logp
        dist_logp = self.distance_logp(amr, alignments, align)
        tokens_count = self.tokens_count[token_label]
        partial_scores = self.naive_subgraph_model.partial_logp(amr, align)
        subgraph_prior = self.naive_subgraph_model.subgraph_prior(amr, align)
        score = self.logp(amr, alignments, align)
        return {'tokens':token_label,
                'subgraph':subgraph,
                'score':score,
                'logP(subgraph|tokens)':logp1,
                'logP(tokens|subgraph)': logp2,
                'logP(tokens)':tokens_logp,
                'logP(subgraph)':subgraph_prior,
                'logP(distance)':dist_logp,
                'partial_logp':partial_scores,
                'tokens_count': tokens_count,
                }

class Naive_Subgraph_Model:

    def __init__(self, tokens_count, tokens_total, alpha):

        self.alpha = alpha
        self.concept_translation_count = {}
        self.concept_translation_total = 0
        self.edge_translation_count = {}
        self.edge_translation_total = 0

        self.tokens_count = tokens_count
        self.tokens_total = tokens_total

        self.concept_count = Counter()
        self.concept_total = 0
        self.edge_count = Counter()
        self.edge_total = 0

        self.tokens_rank = {t:i+1 for i,t in enumerate(sorted(self.tokens_count, key=lambda x:self.tokens_count[x], reverse=True))}

    def update_parameters(self, amrs):

        self.null_count = len(amrs) # this is kind of a hack

        for amr in amrs:
            # concept stats
            concepts = [amr.nodes[n] for n in amr.nodes]
            concepts = [c.replace(' ', '_') for c in concepts]
            edges = [f'({amr.nodes[s]},{r},{amr.nodes[t]})' for s,r,t in amr.edges]
            edges = [e.replace(' ','_') for e in edges]
            all_tokens = [' '.join(amr.lemmas[t] for t in span) for span in amr.spans]
            for token_label in set(all_tokens):
                if token_label not in self.concept_translation_count:
                    self.concept_translation_count[token_label] = Counter()
                    self.edge_translation_count[token_label] = Counter()
                for label in set(concepts):
                    self.concept_translation_count[token_label][label] += 1
                for label in set(edges):
                    self.edge_translation_count[token_label][label] += 1
            for label in set(concepts):
                self.concept_count[label]+=1
            for label in set(edges):
                self.edge_count[label]+=1

        self.concept_total = sum(self.concept_count[l]+self.alpha for l in self.concept_count)
        self.edge_total = sum(self.edge_count[l] + self.alpha for l in self.edge_count)
        self.concept_translation_total = \
            sum(self.concept_translation_count[t][p] for t in self.concept_translation_count for p in self.concept_translation_count[t])
        self.concept_translation_total += self.alpha*len(self.tokens_count)*len(self.concept_count)
        self.edge_translation_total = \
            sum(self.edge_translation_count[t][p] for t in self.edge_translation_count for p in self.edge_translation_count[t])
        self.edge_translation_total += self.alpha * len(self.tokens_count)*len(self.edge_count)

    def logp(self, amr, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        if not align.nodes:
            return self.null_logp(token_label)

        partial_logp = self.partial_logp(amr, align)
        logp_subgraph_for_tokens = sum(partial_logp.values())

        token_logp = math.log(self.tokens_count[token_label] + self.alpha) - math.log(self.tokens_total)

        subgraph_logp = self.subgraph_prior(amr, align)
        logp_tokens_for_subgraph = logp_subgraph_for_tokens + token_logp - subgraph_logp

        return logp_subgraph_for_tokens + logp_tokens_for_subgraph

    def concept_logp(self, token_label, concept_label):
        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        joint_logp = math.log(self.concept_translation_count[token_label][concept_label] + self.alpha) \
                         - math.log(self.concept_translation_total)
        logp = joint_logp - token_logp
        return logp

    def edge_conditional_logp(self, token_label, edge_label, source_label):

        token_logp = math.log(self.tokens_count[token_label] + self.alpha) - math.log(self.tokens_total)
        joint_logp = math.log(self.edge_translation_count[token_label][edge_label] + self.alpha) \
                     - math.log(self.edge_translation_total)
        source_logp = self.concept_logp(token_label, source_label)
        logp = joint_logp - token_logp - source_logp
        return logp

    def null_logp(self, token_label):
        if token_label in self.tokens_rank:
            rank = self.tokens_rank[token_label]
        else:
            rank = len(self.tokens_rank)
        p = 1 / (rank)
        logp = 0.5*math.log(p)
        return max(logp, math.log(0.05)) - math.log(len(self.tokens_count))

    def subgraph_prior(self, amr, align):
        concept_labels, root_labels, edge_labels, source_labels = self.get_partial_alignment_labels(amr, align.nodes)
        prior = 0
        count = 0
        if not concept_labels:
            return math.log(self.null_count) - math.log(self.concept_total)
        else:
            for label in root_labels:
                prior += math.log(self.concept_count[label] + self.alpha) \
                             - math.log(self.concept_total)
                count += 1
            for edge, source in zip(edge_labels, source_labels):
                prior += math.log(self.edge_count[edge] + self.alpha) - math.log(self.edge_total)
                prior -= math.log(self.concept_count[source] + self.alpha) \
                             - math.log(self.concept_total)
                count+=1
        return prior

    def get_partial_alignment_labels(self, amr, nodes):
        if not nodes:
            return [], [], [], []
        if len(nodes)==1:
            concept = amr.nodes[nodes[0]]
            concept = concept.replace(' ','_')
            return [concept], [concept], [], []
        edges = [(s, r, t) for s, r, t in amr.edges if s in nodes and t in nodes]
        roots = [n for n in nodes if not any(n==t for s,r,t in edges)]
        # nodes, edges = self._remove_ignored_parts(amr, nodes, edges)
        root_labels = [amr.nodes[n] for n in roots]
        root_labels = [s.replace(' ','_') for s in root_labels]
        concept_labels = [amr.nodes[n] for n in nodes]
        concept_labels = [s.replace(' ', '_') for s in concept_labels]
        edge_labels = [f'({amr.nodes[s]},{r},{amr.nodes[t]})' for s, r, t in edges]
        edge_labels = [s.replace(' ', '_') for s in edge_labels]
        source_labels = [amr.nodes[s].replace('"','') for s, r, t in edges]
        return concept_labels, root_labels, edge_labels, source_labels

    def partial_logp(self, amr, align):
        # joint probability of subgraph parts p(n1)p(edge=(n1,r1,n2))p(edge=(n1,r2,n3))...
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        concept_labels, root_labels, edge_labels, source_labels = self.get_partial_alignment_labels(amr, align.nodes)
        parts = {}
        if not concept_labels:
            parts['Null'] = self.null_logp(token_label)
        else:
            for label in root_labels:
                p = label
                i = 0
                while p in parts:
                    i += 1
                    p = f'{label}:{i}'
                parts[p] = self.concept_logp(token_label, label)
            for edge, source in zip(edge_labels, source_labels):
                p = edge
                i = 0
                while p in parts:
                    i += 1
                    p = f'{edge}:{i}'
                parts[p] = self.edge_conditional_logp(token_label, edge, source)
        return parts


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

    for i in range(iters-1):
        print(f'Epoch {i}')
        alignments = align_all(align_model, amrs)
        align_model.update_parameters(amrs, alignments)

        Display.style(amrs[:100], amr_file.replace('.txt', '') + f'.subgraphs.no-pretrain{i}.html')
    print(f'Epoch {iters-1}')
    alignments = align_all(align_model, amrs)
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