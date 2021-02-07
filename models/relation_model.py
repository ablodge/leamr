import math
from collections import Counter
from statistics import mean, stdev

from amr_utils.alignments import AMR_Alignment

from evaluate.utils import coverage
from models.base_model import Alignment_Model
from models.distance_model import Gaussian_Distance_Model, Skellam_Distance_Model
from models.naive_model import Concept_Edge_Model
from models.null_model import Null_Model
from rule_based.relation_rules import rule_based_anchor_relation, rule_based_align_relations, exact_match_relations, \
    normalize_relation, english_ignore_tokens

ENGLISH = True

class Relation_Model(Alignment_Model):

    def __init__(self, amrs, subgraph_alignments, alpha=1):
        super().__init__(amrs, alpha)

        self.distance_model_parent = Skellam_Distance_Model(mean=-1, stdev=2.)
        self.distance_model_child = Skellam_Distance_Model(mean=+1, stdev=2.)
        self.null_model = Null_Model(self)

        self.subgraph_alignments = subgraph_alignments

        self.edges_count = Counter()
        self.edges_total = 0

        self.concept_edge_model = Concept_Edge_Model(mode='relation')
        self.concept_edge_model.update_parameters(amrs)

    def trans_logp(self, amr, alignments, align):

        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        align_label = self.get_alignment_label(amr, align)
        sub_align = amr.get_alignment(self.subgraph_alignments, token_id=align.tokens[0])

        if not align.edges and not sub_align.nodes:
            return self.null_model.logp(amr, token_label, align.tokens[0])
        elif token_label in self.translation_count and align_label in self.translation_count[token_label]:
            trans_logp = super().logp(amr, alignments, align)
        else:
            trans_logp = self.concept_edge_model.factorized_logp(amr, align)
        return trans_logp

    def inductive_bias(self, amr, alignments, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        subgraph_label = self.get_alignment_label(amr, align)
        if not align.nodes:
            inductive_bias = self.null_model.inductive_bias(token_label)
        else:
            inductive_bias = self.concept_edge_model.inductive_bias(amr, align, subgraph_label)
        return 0 #inductive_bias

    def logp(self, amr, alignments, align):
        trans_logp = self.trans_logp(amr, alignments, align)
        inductive_bias = self.inductive_bias(amr, alignments, align)
        dist_logp = self.distance_logp(amr, alignments, align)

        return trans_logp + dist_logp #+ inductive_bias

    def get_alignment_label(self, amr, align):
        sub_align = amr.get_alignment(self.subgraph_alignments, token_id=align.tokens[0])
        if not align.edges:
            if sub_align.nodes:
                return '<no_args>'
            else:
                return '<null>'
        internal_nodes = [n for n in sorted(sub_align.nodes)]
        internal_nodes = {n:f'n{i}' for i,n in enumerate(internal_nodes)}
        new_edges = []
        for e in align.edges:
            e = normalize_relation(e)
            s,r,t = normalize_relation(e)
            s = internal_nodes[s] if s in internal_nodes else 'v'
            t = internal_nodes[t] if t in internal_nodes else 'v'
            new_edges.append(f'{s}_{r}_{t}')
        label = [e for e in sorted(new_edges)]
        label = ' '.join(label)
        return label

    def distance_logp(self, amr, alignments, align):

        default = (self.distance_model_parent.logp(self.distance_model_parent.distance_stdev)
                    + self.distance_model_child.logp(self.distance_model_child.distance_stdev))/2

        external_nodes = []
        sub_align = amr.get_alignment(self.subgraph_alignments, token_id=align.tokens[0])
        for s,r,t in align.edges:
            if s not in sub_align.nodes:
                external_nodes.append(s)
            if t not in sub_align.nodes:
                external_nodes.append(t)

        if not external_nodes:
            return default

        parent_dists = []
        child_dists = []
        for s, r, t in align.edges:
            salign = amr.get_alignment(self.subgraph_alignments, node_id=s)
            talign = amr.get_alignment(self.subgraph_alignments, node_id=t)
            if salign:
                dist = self.distance_model_parent.distance(amr, salign.tokens, align.tokens)
                if dist!=0:
                    parent_dists.append(dist)
            if talign:
                dist = self.distance_model_child.distance(amr, talign.tokens, align.tokens)
                if dist!=0:
                    t_parents = [s2 for s2, r2, t2 in amr.edges if t2 == t]
                    # ignore reentrancies
                    if len(t_parents) > 1:
                        reentrancy = False
                        for s2 in t_parents:
                            if s == s2: continue
                            s2_align = amr.get_alignment(self.subgraph_alignments, node_id=s2)
                            other_dist = self.distance_model_child.distance(amr, talign.tokens, s2_align.tokens)
                            if other_dist <= dist:
                                reentrancy = True
                                break
                        if reentrancy:
                            continue
                    child_dists.append(dist)
        parent_logp = 0
        child_logp = 0
        if parent_dists:
            dist = min(parent_dists)
            parent_logp = self.distance_model_parent.logp(dist)
        if child_dists:
            dist = sum(child_dists)/len(child_dists)
            child_logp = self.distance_model_child.logp(dist)

        if parent_logp!=0 and child_logp!=0:
            return (parent_logp+child_logp)/2
        elif parent_logp!=0:
            return parent_logp
        elif child_logp!=0:
            return child_logp
        else:
            return default

    def get_unaligned(self, amr, alignments):
        unaligned = set(amr.edges)
        for align in alignments[amr.id]:
            # if align.type!='relation': continue
            for e in align.edges:
                if e in unaligned:
                    unaligned.remove(e)
        return list(unaligned)

    def update_parameters(self, amrs, relation_alignments):
        super().update_parameters(amrs, relation_alignments)

        for amr in amrs:
            if amr.id not in relation_alignments:
                continue
            for align in relation_alignments[amr.id]:
                align_label = self.get_alignment_label(amr, align)
                self.edges_count[align_label] += 1

        self.edges_total = sum(self.edges_count[e] for e in self.edges_count)
        self.edges_total += self.alpha * len(self.edges_count)

        distances1 = []
        distances2 = []

        for amr in amrs:
            if amr.id not in relation_alignments:
                continue
            # distance stats
            for align in relation_alignments[amr.id]:
                for s, r, t in align.edges:
                    sa = amr.get_alignment(self.subgraph_alignments, node_id=s)
                    ta = amr.get_alignment(self.subgraph_alignments, node_id=t)
                    if sa:
                        dist = self.distance_model_parent.distance(amr, sa.tokens, align.tokens)
                        if dist!=0:
                            distances1.append(dist)
                    if ta:
                        dist = self.distance_model_parent.distance(amr, ta.tokens, align.tokens)
                        if dist != 0:
                            distances2.append(dist)

        distance_mean1 = mean(distances1)
        distance_stdev1 = stdev(distances1)
        self.distance_model_parent.update_parameters(distance_mean1, distance_stdev1)
        distance_mean2 = mean(distances2)
        distance_stdev2 = stdev(distances2)
        self.distance_model_child.update_parameters(distance_mean2, distance_stdev2)

    def get_initial_alignments(self, amrs, preprocess=True):

        relation_alignments = {}
        for j, amr in enumerate(amrs):
            print(f'\r{j} / {len(amrs)} preprocessed', end='')
            relation_alignments[amr.id] = []
            for span in amr.spans:
                relation_alignments[amr.id].append(AMR_Alignment(type='relation', tokens=span, amr=amr))
            rule_based_align_relations(amr, self.subgraph_alignments, relation_alignments)
            exact_match_relations(amr, self.subgraph_alignments, relation_alignments)
        print('\r', end='')
        print('Preprocessing coverage:', coverage(amrs, relation_alignments, mode='edges'))
        return relation_alignments


    def readable_logp(self, amr, alignments, align):
        readable = super().readable_logp(amr, alignments, align)
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        tokens_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        dist_logp = self.distance_logp(amr, alignments, align)

        trans_logp = self.trans_logp(amr, alignments, align)
        inductive_bias = self.inductive_bias(amr, alignments, align)

        subgraph_align = amr.get_alignment(self.subgraph_alignments, token_id=align.tokens[0])
        rel_type = 'source' if any(s in subgraph_align.nodes for s,r,t in align.edges) \
            else 'target' if any(t in subgraph_align.nodes for s,r,t in align.edges) \
            else 'null' if not align.edges \
            else 'function_word'
        readable.update(
            {'relations':self.get_alignment_label(amr, align),
             'logP(rels|tokens)':trans_logp,
             'logP(tokens|rels)': inductive_bias,
             'logP(distance)':dist_logp,
             'rel_type':rel_type,
             }
        )
        return readable

    def align(self, amr, relation_alignments, e, unaligned=None, return_all=False):
        # get candidates
        candidate_spans = [align.tokens for align in self.subgraph_alignments[amr.id] if not align.nodes]
        candidate_spans = [span for span in candidate_spans if not amr.get_alignment(relation_alignments, token_id=span[0])]
        candidate_spans = [span for span in candidate_spans if not english_ignore_tokens(amr, span)]
        candidate_neighbors = rule_based_anchor_relation(e)

        readable = []
        scores1 = {}
        aligns1 = {}
        for i, span in enumerate(candidate_spans):
            new_align = AMR_Alignment(type='relation', tokens=span, edges=[e], amr=amr)
            replaced_align = AMR_Alignment(type='relation', tokens=span, edges=[], amr=amr)
            scores1[i] = self.logp(amr, relation_alignments, new_align) - self.logp(amr, relation_alignments, replaced_align)
            scores1[i] += self.inductive_bias(amr, relation_alignments, new_align) - self.inductive_bias(amr, relation_alignments, replaced_align)
            aligns1[i] = new_align
            # readable.append(self.readable_logp(amr,alignments, new_align))
        scores2 = {}
        aligns2 = {}
        for i, neighbor in enumerate(candidate_neighbors):
            sub_align = amr.get_alignment(self.subgraph_alignments, node_id=neighbor)
            span = sub_align.tokens
            if not span: continue
            if span not in amr.spans:
                raise Exception('Subgraph Alignment has Faulty Span:', span)
            replaced_align = amr.get_alignment(relation_alignments, token_id=span[0])
            new_align = AMR_Alignment(type='relation', tokens=replaced_align.tokens, edges=replaced_align.edges+[e], amr=amr)
            scores2[i] = self.logp(amr, relation_alignments, new_align) - self.logp(amr, relation_alignments, replaced_align)
            scores2[i] += self.inductive_bias(amr, relation_alignments, new_align) - self.inductive_bias(amr, relation_alignments, replaced_align)
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

        if return_all:
            return all_aligns, all_scores

        best_span = max(all_scores.keys(), key=lambda x:all_scores[x])
        best_score = all_scores[best_span]
        best_align = all_aligns[best_span]

        # readable = [r for r in sorted(readable, key=lambda x:x['score'], reverse=True)]
        return best_align, best_score
