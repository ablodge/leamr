import math
from collections import Counter
from statistics import mean, stdev

from amr_utils.alignments import AMR_Alignment

from evaluate.utils import coverage
from models.base_model import Alignment_Model
from models.distance_model import Skellam_Distance_Model

ENGLISH = True

PRAGMATIC_RATE = 0.01

class Reentrancy_Model(Alignment_Model):

    def __init__(self, amrs, subgraph_alignments, relation_alignments, alpha=1):
        super().__init__(amrs, alpha)

        self.distance_model_parent = Skellam_Distance_Model()
        self.distance_model_child = Skellam_Distance_Model()

        self.subgraph_alignments = subgraph_alignments
        self.relation_alignments = relation_alignments

        self.edges_count = Counter()
        self.edges_total = 0

        self.allowed_types_memo_ = None

        edge_labels = set()
        for amr in amrs:
            parents = {s:[] for s in amr.nodes}
            children = {t:[] for t in amr.nodes}
            taken_tokens = set()
            for s,r,t in amr.edges:
                parents[s].append((s,r,t))
                children[t].append((s,r,t))
            for align in subgraph_alignments[amr.id]:
                token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
                for n in align.nodes:
                    for e in parents[n]:
                        s,r,t = e
                        if t in align.nodes: continue
                        partial_align = AMR_Alignment(type='relation', tokens=align.tokens, edges=[e])
                        edge_label = self.get_alignment_label(amr, partial_align)
                        edge_labels.add(edge_label)
                    for e in children[n]:
                        s, r, t = e
                        if s in align.nodes: continue
                        partial_align = AMR_Alignment(type='relation', tokens=align.tokens, edges=[e])
                        edge_label = self.get_alignment_label(amr, partial_align)
                        edge_labels.add(edge_label)
                if not align.nodes and token_label not in taken_tokens:
                    edges = set()
                    for e in amr.edges:
                        partial_align = AMR_Alignment(type='relation', tokens=align.tokens, edges=[e])
                        edge_label = self.get_alignment_label(amr, partial_align)
                        edges.add(edge_label)
                    taken_tokens.add(token_label)

    def trans_logp(self, amr, alignments, align):

        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        align_label = self.get_alignment_label(amr, align)

        if not align.edges:
            raise Exception('Faulty Alignment, align')
        elif token_label in self.translation_count and align_label in self.translation_count[token_label]:
            trans_logp = super().logp(amr, alignments, align)
        else:
            trans_logp = math.log(1e-6)
        return trans_logp

    def logp(self, amr, alignments, align):
        trans_logp = self.trans_logp(amr, alignments, align)
        dist_logp = self.distance_logp(amr, alignments, align)

        return trans_logp + dist_logp

    def get_alignment_label(self, amr, align):
        # coordination
        # comparative control
        # pronominal coreference
        # repetition coreference
        # adjunct control
        # unmarked adjunct control
        # control
        # pragmatic
        type = align.type.split(':')[-1]
        if type in ['primary','repetition','coref']:
            label = f'{type}'
        else:
            # lemma = ' '.join(amr.lemmas[t] for t in align.tokens)
            rel = align.edges[0][1]
            label = f'{type}:{rel}'
        return label


    def distance_logp(self, amr, alignments, align):

        parent_dist = 0
        child_dist = 0
        for s, r, t in align.edges:
            salign = amr.get_alignment(self.subgraph_alignments, node_id=s)
            talign = amr.get_alignment(self.subgraph_alignments, node_id=t)
            if salign:
                parent_dist = self.distance_model_parent.distance(amr, salign.tokens, align.tokens)
            if talign:
                child_dist = self.distance_model_child.distance(amr, talign.tokens, align.tokens)
        parent_logp = self.distance_model_parent.logp(parent_dist)
        child_logp = self.distance_model_child.logp(child_dist)

        return parent_logp+child_logp

    def get_unaligned(self, amr, alignments):
        if not hasattr(amr, 'reentrancies'):
            amr.reentrancies = [e for e in amr.edges if len([e2 for e2 in amr.edges if e2[-1]==e[-1]])>1]
        unaligned = {e for e in amr.reentrancies}
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

    def align_primary_edges(self, amr, alignments):
        if not hasattr(amr, 'reentrancies'):
            amr.reentrancies = [e for e in amr.edges if len([e2 for e2 in amr.edges if e2[-1]==e[-1]])>1]
        ts = {t for s,r,t in amr.reentrancies}
        for t in ts:
            candidates = [e for e in amr.reentrancies if e[-1]==t]
            talign = amr.get_alignment(self.subgraph_alignments, node_id=t)
            rel_align = amr.get_alignment(self.relation_alignments, token_id=talign.tokens[0])
            if rel_align and any(e in rel_align.edges for e in candidates):
                span = talign.tokens
                e = [e for e in candidates if e in rel_align.edges][0]
            else:
                dists = {}
                for s,r,t in candidates:
                    if not amr.get_alignment(self.relation_alignments, edge=(s,r,t)):
                        continue
                    salign = amr.get_alignment(self.subgraph_alignments, node_id=s)
                    talign = amr.get_alignment(self.subgraph_alignments, node_id=t)
                    dist = self.distance_model_parent.distance(amr, salign.tokens, talign.tokens)
                    dists[(s,r,t)] = (abs(dist), salign.tokens[0])
                e = min(dists, key=lambda x:dists[x])
                ealign = amr.get_alignment(self.relation_alignments, edge=e)
                span = ealign.tokens
            if not span:
                continue
            alignments[amr.id].append(AMR_Alignment(type='reentrancy:primary', tokens=span, edges=[e]))

    def get_initial_alignments(self, amrs, preprocess=True):

        reentrancy_alignments = {}
        for j, amr in enumerate(amrs):
            print(f'\r{j} / {len(amrs)} preprocessed', end='')
            reentrancy_alignments[amr.id] = []
            self.align_primary_edges(amr, reentrancy_alignments)
        print('\r', end='')
        print('Preprocessing coverage:', coverage(amrs, reentrancy_alignments, mode='edges'))
        return reentrancy_alignments


    def readable_logp(self, amr, alignments, align):
        readable = super().readable_logp(amr, alignments, align)
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        tokens_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        dist_logp = self.distance_logp(amr, alignments, align)

        trans_logp = self.trans_logp(amr, alignments, align)
        e = align.edges[0]
        subgraph_align = amr.get_alignment(self.subgraph_alignments, token_id=align.tokens[0])
        readable.update(
            {'relations':self.get_alignment_label(amr, align),
             'source': amr.nodes[e[0]],
             'target': amr.nodes[e[-1]],
             'logP(rels|tokens)':trans_logp,
             'logP(distance)':dist_logp,
             }
        )
        return readable

    def get_allowed_types(self, amr):
        if self.allowed_types_memo_ is not None and amr.id==self.allowed_types_memo_[0]:
            return self.allowed_types_memo_[1]
        allowed_types = {}
        for e in amr.reentrancies:
            allowed_types[e] = {}
            rel_align = amr.get_alignment(self.relation_alignments, edge=e)
            s_align = amr.get_alignment(self.subgraph_alignments, node_id=e[0])
            t_align = amr.get_alignment(self.subgraph_alignments, node_id=e[-1])
            neighbors = [(s, r, t) for s, r, t in amr.reentrancies if t == e[-1] and (s, r, t) != e]

            for i,span in enumerate(amr.spans):
                # types = ['coordination','comparative','coref','repetition','adjunct1','adjunct2','control','pragmatic']
                pos = amr.pos[span[0]]
                lemma = ' '.join(amr.lemmas[t] for t in span).lower()

                span_types = []
                # coref style alignments
                sub_align = amr.get_alignment(self.subgraph_alignments, token_id=span[0])
                if not sub_align:
                    # coref
                    if pos in ['PRP', 'PRP$', 'WP']:
                        for corefs in amr.coref:
                            if span in corefs and t_align.tokens in corefs:
                                span_types.append('coref')
                        if not any(span in corefs for corefs in amr.coref):
                            span_types.append('coref')
                    # repetition
                    repetitions = [span for span in amr.spans if
                                   ' '.join(amr.lemmas[t] for t in span).lower()[:6] == lemma[:6]]
                    if len(repetitions) > 1 and t_align.tokens in repetitions:
                        span_types.append('repetition')
                # control style alignemnts
                elif span!=rel_align.tokens:
                    # coordination
                    grandparents = {s for s, r, t in amr.edges if t == e[0]}
                    for e2 in neighbors:
                        e2_grandparents = {s for s, r, t in amr.edges if t == e2[0]}
                        coord = [n for n in grandparents if amr.nodes[n] in ['and', 'or'] and n in e2_grandparents]
                        for c in coord:
                            if span == amr.get_alignment(self.subgraph_alignments, node_id=c).tokens:
                                span_types.append('coordination')
                    # control
                    content_words = ['VB', 'VBD', 'VBZ', 'VBG', 'VBP', 'VBN', 'NN', 'NNS', 'JJ', 'JJR', 'JJS']
                    if pos in content_words and amr.pos[s_align.tokens[0]] in content_words and span[0] < \
                            s_align.tokens[0]:
                        for e2 in neighbors:
                            e2_align = amr.get_alignment(self.relation_alignments, edge=e2)
                            s2_align = amr.get_alignment(self.subgraph_alignments, node_id=e2[0])
                            if span == e2_align.tokens and any(s in s2_align.nodes and t in s_align.nodes for s, r, t in e2_align.edges):
                                span_types.append('control')
                    # adjunct1
                    next_span = [s for s in amr.spans if s[0] > span[0]]
                    next_span = next_span[0] if next_span else None
                    next_pos = amr.pos[next_span[0]] if next_span else None
                    if (pos == 'IN' and next_pos == 'VBG') or (lemma == 'to' and next_pos == 'VB' and e[1] == ':purpose'):
                        if next_span == rel_align.tokens:
                            span_types.append('adjunct control')
                # other
                else:
                    # comparative
                    parent = amr.nodes[e[0]]
                    if parent == 'have-degree-91':
                        span_types.append('comparative')
                    # adjunct2
                    if pos=='VBG':
                        rel_align = amr.get_alignment(self.relation_alignments, edge=e)
                        if span == rel_align.tokens:
                            span_types.append('unmarked adjunct control')
                    # pragmatic
                    rel_align = amr.get_alignment(self.relation_alignments, edge=e)
                    if span==rel_align.tokens:
                        span_types.append('pragmatic')
                allowed_types[e][tuple(span)] = span_types
        self.allowed_types_memo_ = amr.id, allowed_types
        return allowed_types

    def align(self, amr, reentrancy_alignments, e, unaligned=None, return_all=False):
        # get candidates
        allowed_types = self.get_allowed_types(amr)
        candidate_spans = [span for span in amr.spans if allowed_types[e][tuple(span)]]
        # candidate_spans = [span for span in candidate_spans if not amr.get_alignment(reentrancy_alignments, token_id=span[0])]
        candidate_neighbors = [] #[e]
        neighbor_aligns = [amr.get_alignment(reentrancy_alignments, edge=(s,r,t)) for s,r,t in amr.reentrancies if t==e[-1] and e!=(s,r,t)]
        # if all(a.type!='reentrancy:primary' for a in neighbor_aligns):
        #     candidate_spans = []

        readable = []
        scores1 = {}
        aligns1 = {}
        for i, span in enumerate(candidate_spans):
            type = allowed_types[e][tuple(span)][0]
            new_align = AMR_Alignment(type=f'reentrancy:{type}', tokens=span, edges=[e], amr=amr)
            # replaced_align = AMR_Alignment(type='relation', tokens=span, edges=[], amr=amr)
            scores1[i] = self.logp(amr, reentrancy_alignments, new_align) #- self.logp(amr, reentrancy_alignments, replaced_align)
            # scores1[i] = self.inductive_bias(amr, reentrancy_alignments, new_align) - self.inductive_bias(amr, reentrancy_alignments, replaced_align)
            if type=='pragmatic':
                scores1[i] += math.log(PRAGMATIC_RATE)
            aligns1[i] = new_align
            # readable.append(self.readable_logp(amr,alignments, new_align))
        scores2 = {}
        aligns2 = {}
        for i, neighbor in enumerate(candidate_neighbors):
            rel_align = amr.get_alignment(self.relation_alignments, edge=neighbor)
            span = rel_align.tokens
            if not span: continue
            if span not in amr.spans:
                raise Exception('Relation Alignment has Faulty Span:', span)
            new_align = AMR_Alignment(type='reentrancy:primary', tokens=rel_align.tokens, edges=[e], amr=amr)
            scores2[i] = self.logp(amr, reentrancy_alignments, new_align) #- self.logp(amr, reentrancy_alignments, replaced_align)
            # scores2[i] = self.inductive_bias(amr, reentrancy_alignments, new_align) - self.inductive_bias(amr, reentrancy_alignments, replaced_align)
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
