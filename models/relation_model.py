import math
from collections import Counter
from statistics import mean, stdev

from amr_utils.alignments import AMR_Alignment

from models.alignment_model import Alignment_Model
from models.distance_model import Gaussian_Distance_Model
from rule_based.relation_rules import rule_based_anchor_relation, rule_based_align_relations, exact_match_relations

ENGLISH = True

class Relation_Model(Alignment_Model):

    def __init__(self, amrs, subgraph_alignments, alpha=1):
        super().__init__(amrs, alpha)

        self.distance_model_parent = Gaussian_Distance_Model()
        self.distance_model_child = Gaussian_Distance_Model()

        self.subgraph_alignments = subgraph_alignments

        self.simple_translation_count = {}
        self.simple_translation_total = 0
        edge_labels = set()
        for amr in amrs:
            parents = {s:[] for s in amr.nodes}
            children = {t:[] for t in amr.nodes}
            for s,r,t in amr.edges:
                parents[s].append((s,r,t))
                children[t].append((s,r,t))
                edge_labels.add(self.get_alignment_label(amr,[(s,r,t)]))
            for align in subgraph_alignments[amr.id]:
                token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
                if token_label not in self.simple_translation_count:
                    self.simple_translation_count[token_label] = Counter()
                for n in align.nodes:
                    for e in parents[n]:
                        s,r,t = e
                        if t in align.nodes: continue
                        self.simple_translation_count[token_label][self.get_alignment_label(amr,[e])] += 1
                    for e in children[n]:
                        s, r, t = e
                        if s in align.nodes: continue
                        self.simple_translation_count[token_label][self.get_alignment_label(amr, [e])] += 1
                if not align.nodes:
                    for e in amr.edges:
                        self.simple_translation_count[token_label][self.get_alignment_label(amr, [e])] += 1
        self.simple_translation_total = len(self.tokens_count)*len(edge_labels)*self.alpha


    def logp(self, amr, alignments, align):

        tokens = ' '.join(amr.lemmas[t] for t in align.tokens)
        align_label = self.get_alignment_label(amr, align.nodes)

        if (tokens, align_label) in self._trans_logp_memo or\
                tokens in self.translation_count and align_label in self.translation_count[tokens]:
            trans_logp = super().logp(amr, alignments, align)
        else:
            trans_logp = 0
            for e in align.edges:
                trans_logp += math.log(self.simple_translation_count[tokens][self.get_alignment_label(amr,[e])] + self.alpha) \
                              - math.log(self.simple_translation_total)
                trans_logp -= math.log(self.tokens_count[tokens]+self.alpha) - math.log(self.tokens_total)

        dist_logp = self.distance_logp(amr, alignments, align)

        return trans_logp + dist_logp

    def get_alignment_label(self, amr, edges):
        if not edges:
            return 'Null'
        # nodes = [t for s,r,t in edges]+[s for s,r,t in edges]
        # nodes = [n for n in sorted(set(nodes))]
        # edges = [(nodes.index(s),r,nodes.index(t)) for s,r,t in edges]
        label = [r for s,r,t in sorted(edges)]
        label = ' '.join(label)
        return label

    def distance_logp(self, amr, alignments, align):
        parent_dists = []
        child_dists = []
        for s, r, t in align.edges:
            salign = amr.get_alignment(self.subgraph_alignments, node_id=s)
            talign = amr.get_alignment(self.subgraph_alignments, node_id=t)
            if salign:
                dist = self.distance_model_parent.distance(amr, salign.tokens, align.tokens)
                if dist!=0:
                    parent_dists.append(dist)
            elif talign:
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

        if parent_logp == 0:
            parent_logp = child_logp
        elif child_logp == 0:
            child_logp = parent_logp
        return parent_logp + child_logp

    def get_unaligned(self, amr, alignments):
        unaligned = set(amr.edges)
        for align in alignments[amr.id]:
            if align.type!='relation': continue
            for e in align.edges:
                if e in unaligned:
                    unaligned.remove(e)
        return list(unaligned)

    def update_parameters(self, amrs, relation_alignments):
        super().update_parameters(amrs, relation_alignments)

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
        return relation_alignments


    def readable_logp(self, amr, alignments, align):
        readable = super().readable_logp(amr, alignments, align)
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        tokens_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        dist_logp = self.distance_logp(amr, alignments, align)

        logp = self.logp(amr, alignments, align)-dist_logp
        subgraph_align = amr.get_alignment(self.subgraph_alignments, token_id=align.tokens[0])
        rel_type = 'source' if any(s in subgraph_align.nodes for s,r,t in align.edges) \
            else 'target' if any(t in subgraph_align.nodes for s,r,t in align.edges) \
            else 'null' if not align.edges \
            else 'function_word'
        readable.update(
            {'relations':self.get_alignment_label(amr, align.edges),
             'logP(rels|tokens)':logp,
             'logP(tokens)':tokens_logp,
             'logP(distance)':dist_logp,
             'rel_type':rel_type,
             }
        )
        return readable

    def align(self, amr, relation_alignments, e, unaligned=None):
        # get candidates
        candidate_spans = [align.tokens for align in relation_alignments[amr.id] if not align.nodes]
        candidate_neighbors = rule_based_anchor_relation(e)

        readable = []
        scores1 = {}
        aligns1 = {}
        for i, span in enumerate(candidate_spans):
            new_align = AMR_Alignment(type='relation', tokens=span, edges=[e], amr=amr)
            replaced_align = AMR_Alignment(type='relation', tokens=span, edges=[], amr=amr)
            scores1[i] = self.logp(amr, relation_alignments, new_align) - self.logp(amr, relation_alignments, replaced_align)
            aligns1[i] = new_align
            # readable.append(self.readable_logp(amr,alignments, new_align))
        scores2 = {}
        aligns2 = {}
        for i, neighbor in enumerate(candidate_neighbors):
            span = amr.get_alignment(self.subgraph_alignments, node_id=neighbor).tokens
            replaced_align = amr.get_alignment(relation_alignments, token_id=span[0])
            new_align = AMR_Alignment(type='relation', tokens=replaced_align.tokens, edges=replaced_align.edges+[e], amr=amr)
            scores2[i] = self.logp(amr, relation_alignments, new_align) - self.logp(amr, relation_alignments, replaced_align)
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
