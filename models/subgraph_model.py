import math

from statistics import stdev, mean

from amr_utils.alignments import AMR_Alignment

from evaluate.utils import coverage
from models.base_model import Alignment_Model
from models.distance_model import Gaussian_Distance_Model, Skellam_Distance_Model
from models.naive_model import Node_Model, Internal_Edge_Model, POS_Node_Model
from models.null_model import Null_Model
from rule_based.subgraph_rules import fuzzy_align_subgraphs, postprocess_subgraph, clean_subgraph, clean_alignments, \
    english_is_alignment_forbidden

ENGLISH = True
POS = True

PARTIAL_CREDIT_RATE = 0.1
DUPLICATE_RATE = 0.05


class Subgraph_Model(Alignment_Model):

    def __init__(self, amrs, alpha=0.1, align_duplicates=True):
        # Do not smooth over subgraphs (since the vocabulary is not finite or fixed)
        # The model uses backoff to a partial credit model instead
        super().__init__(amrs, alpha=alpha, smooth_translation=True)
        self.align_duplicates = align_duplicates

        self.distance_model = Skellam_Distance_Model()
        self.null_model = Null_Model(self.tokens_count, self.tokens_total, alpha)
        self.node_model = Node_Model(amrs, alpha=alpha)
        self.edge_model = Internal_Edge_Model(amrs, alpha=alpha)
        if POS:
            self.pos_node_model = POS_Node_Model(amrs, alpha=alpha)
            # self.pos_edge_model = POS_Edge_Model(amrs, alpha=alpha)

        self.is_initialized = False
        self.num_null_aligned = 0

    def trans_logp(self, amr, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        subgraph_label = self.get_alignment_label(amr, align)

        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        if (token_label, subgraph_label) in self._trans_logp_memo:
            # check for memoized answer
            return self._trans_logp_memo[(token_label, subgraph_label)]
        if not align.nodes:
            # null alignment
            trans_logp = self.null_model.logp(amr, token_label, align.tokens[0])
            # if POS:
            #     trans_logp += self.pos_null_model.logp(amr.pos[align.tokens[0]])
            return trans_logp
        elif token_label in self.translation_count and self.translation_count[token_label][subgraph_label]>0:
            # attested alignment
            trans_logp = math.log(self.translation_count[token_label][subgraph_label] + self.alpha) - math.log(self.translation_total)
            trans_logp -= token_logp
        elif len(token_label.split())>1 and any(t in self.translation_count and subgraph_label in self.translation_count[t]
                                                for t in token_label.split()):
            # partial match by token
            max_logp = float('-inf')
            for tok in token_label.split():
                if tok not in self.translation_count: continue
                if self.translation_count[tok][subgraph_label] == 0: continue
                logp = math.log(self.translation_count[tok][subgraph_label] + self.alpha) - math.log(self.translation_total)
                logp -= token_logp
                if logp > max_logp:
                    max_logp = logp
            trans_logp = max_logp + math.log(PARTIAL_CREDIT_RATE)
        else:
            # partial match by subgraph parts
            trans_logp = self.factorized_logp(amr, align) + math.log(PARTIAL_CREDIT_RATE)
        # if POS:
        #     trans_logp += sum(self.pos_node_model.factorized_logp(amr, align).values())\
        #                   +sum(self.pos_edge_model.factorized_logp(amr, align).values())

        self._trans_logp_memo[(token_label, subgraph_label)] = trans_logp

        return trans_logp

    def factorized_logp(self, amr, align):
        parts = self.node_model.factorized_logp(amr, align)
        parts.update(self.edge_model.factorized_logp(amr, align))
        return sum(parts.values())

    def inductive_bias(self, amr, align):
        bias = 0
        if POS:
            parts = self.pos_node_model.inductive_bias(amr, align)
            # parts.update(self.pos_edge_model.inductive_bias(amr, align))
            bias += sum(parts.values())/len(parts)
        # if not align.nodes:
        #     token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        #     logp = self.null_model.inductive_bias(amr, token_label, align.tokens[0])
        #     if POS:
        #         logp += self.pos_null_model.inductive_bias(amr.pos[align.tokens[0]])
        #     return logp
        # parts = self.node_model.inductive_bias(amr, align)
        # parts.update(self.edge_model.inductive_bias(amr, align))
        # bias = sum(parts.values())/len(parts)
        # if POS:
        #     parts = self.pos_node_model.inductive_bias(amr, align)
        #     parts.update(self.pos_edge_model.inductive_bias(amr, align))
        #     bias += sum(parts.values())/len(parts)
        return bias

    def logp(self, amr, alignments, align, postprocess=True):
        if postprocess:
            postprocess_subgraph(amr, alignments, align, english=ENGLISH)
            align = clean_subgraph(amr, alignments, align)
        if align is None: return float('-inf')

        trans_logp = self.trans_logp(amr, align)
        dist_logp = self.distance_logp(amr, alignments, align)
        inductive_bias = self.inductive_bias(amr, align)
        # ln( P(subgraph|tokens)*P(tokens|subgraph)*P(distance) )
        return trans_logp + dist_logp + inductive_bias

    def get_alignment_label(self, amr, align):

        nodes = align.nodes
        if not nodes:
            return None
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
        if not align.nodes:
            return self.distance_model.logp(self.distance_model.distance_stdev)

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
        n = 0
        if parent_dists:
            dist = min(parent_dists, key=lambda x:abs(x))
            logp += self.distance_model.logp(dist)
            n+=1
        for dist in child_dists:
            logp += self.distance_model.logp(dist)
            n+=1
        if n < 1:
            return self.distance_model.logp(self.distance_model.distance_stdev)
        return logp/n


    def get_initial_alignments(self, amrs, preprocess=True):
        print(f'Apply Rules = {preprocess}')
        alignments = {}
        for j, amr in enumerate(amrs):
            print(f'\rPreprocessing: {j} / {len(amrs)}', end='')
            alignments[amr.id] = []
            for span in amr.spans:
                alignments[amr.id].append(AMR_Alignment(type='subgraph', tokens=span, amr=amr))
            if preprocess:
                fuzzy_align_subgraphs(amr, alignments, english=ENGLISH)
                for align in alignments[amr.id]:
                    postprocess_subgraph(amr, alignments, align, english=ENGLISH)
                    test = clean_subgraph(amr, alignments, align)
                    if test is None:
                        align.nodes.clear()
        print('\r', end='')
        print('Preprocessing coverage:', coverage(amrs, alignments))
        return alignments

    def update_parameters(self, amrs, alignments, prune=True):
        super().update_parameters(amrs, alignments)
        self.node_model.update_parameters(amrs, alignments)
        self.edge_model.update_parameters(amrs, alignments)
        if POS:
            self.pos_node_model.update_parameters(amrs, alignments)
            # self.pos_edge_model.update_parameters(amrs, alignments)

        # prune rare alignments
        if prune:
            for token_label in self.translation_count:
                for subgraph_label in list(self.translation_count[token_label].keys()):
                    if self.tokens_count[token_label] == 0:
                        continue
                    if self.translation_count[token_label][subgraph_label]/self.tokens_count[token_label]<=0.01:
                        del self.translation_count[token_label][subgraph_label]
            for token_label in list(self.translation_count.keys()):
                if self.tokens_count[token_label] == 1:
                    del self.translation_count[token_label]

        self.translation_total += self.null_model.smoothing()

        distances = []

        for amr in amrs:
            if amr.id not in alignments:
                continue
            # distance stats
            for s, r, t in amr.edges:
                sa = amr.get_alignment(alignments, node_id=s)
                ta = amr.get_alignment(alignments, node_id=t)
                if sa and ta:
                    dist = self.distance_model.distance(amr, sa.tokens, ta.tokens)
                    distances.append(dist)

        distance_mean = mean(distances)
        distance_stdev = stdev(distances)
        self.distance_model.update_parameters(distance_mean, distance_stdev)

        self.is_initialized = True

        self.num_null_aligned = 0
        for amr in amrs:
            if amr.id not in alignments:
                continue
            for align in alignments[amr.id]:
                if not align.nodes:
                    self.num_null_aligned += 1

    def align(self, amr, alignments, n, unaligned=None, return_all=False):

        # get candidates
        if unaligned is None:
            unaligned = self.get_unaligned(amr, alignments)
        candidate_spans = [align.tokens for align in alignments[amr.id] if not align.nodes]
        tmp_align = AMR_Alignment(type='subgraph', tokens=[0], nodes=[n])
        postprocess_subgraph(amr, alignments, tmp_align, english=ENGLISH)
        candidate_neighbors = [s for s, r, t in amr.edges if t in tmp_align.nodes and s not in unaligned] + \
                              [t for s, r, t in amr.edges if s in tmp_align.nodes and t not in unaligned]
        for n2 in candidate_neighbors[:]:
            nalign = amr.get_alignment(alignments, node_id=n2)
            if not nalign or nalign.type == 'dupl-subgraph':
                candidate_neighbors.remove(n2)

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

        # special rules for multi-sentence, and, or
        if ENGLISH:
            candidate_spans2 = [span for span in candidate_spans if not english_is_alignment_forbidden(amr, span, n)]
            if amr.nodes[n] == 'multi-sentence' and not candidate_spans:
                candidate_spans2 = candidate_spans
            elif amr.nodes[n] == 'and' and not candidate_spans:
                candidate_spans2 = candidate_spans
            candidate_spans = candidate_spans2

        if amr.nodes[n] in ['multi-sentence', 'and', 'or'] and candidate_spans:
            candidate_neighbors = []
        for n2 in candidate_neighbors[:]:
            if amr.nodes[n2] in ['multi-sentence', 'and', 'or'] and candidate_spans:
                candidate_neighbors.remove(n2)
        if len([n2 for n2 in amr.nodes if amr.nodes[n]==amr.nodes[n2]])>1:
            for s, r, t in amr.edges:
                if t==n and amr.nodes[s] in ['include-91', 'same-01', 'instead-of-91', 'resemble-01', 'differ-02', 'and', 'or']:
                    if len([lemma for lemma in amr.lemmas if amr.nodes[n].split('-')[0]==lemma]) >= \
                        len([n2 for n2 in amr.nodes if amr.nodes[n]==amr.nodes[n2]]):
                        break
                    for s2, r2, t2 in amr.edges:
                        if s2 == s and t2 != t and amr.nodes[t2]==amr.nodes[n] and r2.endswith('1'):
                            candidate_spans = []
                            break
                        elif t2 == s and amr.nodes[s2]==amr.nodes[n] and r2.endswith('1-of'):
                            candidate_spans = []
                            break

        candidate_duplicates = []
        for n2 in amr.nodes:
            if amr.nodes[n].isdigit() or '"' in amr.nodes[n]: break
            if n2!=n and amr.nodes[n]==amr.nodes[n2]:
                align = amr.get_alignment(alignments, node_id=n2)
                if align:
                    candidate_duplicates.append(align.tokens)

        readable = []
        scores1 = {}
        aligns1 = {}
        for i, span in enumerate(candidate_spans):
            new_align = AMR_Alignment(type='subgraph', tokens=span, nodes=[n], amr=amr)
            replaced_align = AMR_Alignment(type='subgraph', tokens=span, nodes=[], amr=amr)
            scores1[i] = self.logp(amr, alignments, new_align) - self.logp(amr, alignments, replaced_align)
            # scores1[i] += self.inductive_bias(amr, new_align) - self.inductive_bias(amr, replaced_align)
            aligns1[i] = new_align
            # readable.append(self.readable_logp(amr,alignments, new_align))
        scores2 = {}
        aligns2 = {}
        for i, neighbor in enumerate(candidate_neighbors):
            replaced_align = amr.get_alignment(alignments, node_id=neighbor)
            if replaced_align.type.startswith('dupl'): continue
            new_align = AMR_Alignment(type=replaced_align.type, tokens=replaced_align.tokens, nodes=replaced_align.nodes+[n], amr=amr)
            scores2[i] = self.logp(amr, alignments, new_align) - self.logp(amr, alignments, replaced_align, postprocess=False)
            # scores2[i] += self.inductive_bias(amr, new_align) - self.inductive_bias(amr, replaced_align)
            aligns2[i] = new_align
            # hack
            # if amr.nodes[n] in [amr.nodes[n2] for n2 in replaced_align.nodes]:
            #     scores2[i] += math.log(1/100)
            # readable.append(self.readable_logp(amr, alignments, new_align))
        scores3 = {}
        aligns3 = {}
        if self.align_duplicates:
            for i, span in enumerate(candidate_duplicates):
                new_align = AMR_Alignment(type='dupl-subgraph', tokens=span, nodes=[n], amr=amr)
                replaced_align = amr.get_alignment(alignments, token_id=span[0])
                scores3[i] = math.log(DUPLICATE_RATE) + self.logp(amr, alignments, new_align) - self.logp(amr, alignments, replaced_align, postprocess=False)
                # scores3[i] += self.inductive_bias(amr, new_align) - self.inductive_bias(amr, replaced_align)
                aligns3[i] = new_align

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
        for x in scores3:
            span = tuple(aligns3[x].tokens)
            all_scores[span] = scores3[x]
            all_aligns[span] = aligns3[x]

        if not all_scores:
            return None, None

        if return_all:
            return all_aligns, all_scores

        best_span = max(all_scores.keys(), key=lambda x:all_scores[x])
        best_score = all_scores[best_span]
        best_align = all_aligns[best_span]

        # readable = [r for r in sorted(readable, key=lambda x:x['score'], reverse=True)]
        return best_align, best_score

    def postprocess_alignments(self, amr, alignments):
        clean_alignments(amr, alignments)

    def get_unaligned(self, amr, alignments):
        aligned = set()
        for align in alignments[amr.id]:
            aligned.update(align.nodes)
        return [n for n in amr.nodes if n not in aligned]

    def readable_logp(self, amr, alignments, align):
        readable = super().readable_logp(amr, alignments, align)
        subgraph_label = self.get_alignment_label(amr, align)
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        pos = amr.pos[align.tokens[0]]

        trans_logp = self.trans_logp(amr, align)
        inductive_bias = self.inductive_bias(amr, align)
        dist_logp = self.distance_logp(amr, alignments, align)
        partial = self.node_model.factorized_logp(amr, align)
        partial.update(self.edge_model.factorized_logp(amr, align))
        partial_inductive = self.node_model.inductive_bias(amr, align)
        partial_inductive.update(self.edge_model.inductive_bias(amr, align))
        if not align.nodes:
            subgraph_label = '<null>'
            partial['<null>'] = self.null_model.logp(amr, token_label, align.tokens[0])
            partial_inductive['<null>'] = self.null_model.inductive_bias(amr, token_label, align.tokens[0])

        partial_pos2 = {}
        if POS:
            partial_pos2 = self.pos_node_model.inductive_bias(amr, align)
            # partial_pos2.update(self.pos_edge_model.inductive_bias(amr, align))

        readable.update(
            {'subgraph':subgraph_label,
             'POS':pos,
             'logP(subgraph|tokens)':trans_logp,
             'logP(tokens|subgraph)': inductive_bias,
             'logP(distance)':dist_logp,
             'logP(POS|subgraph)':inductive_bias,
             'factorized logp':partial,
             # 'factorized inductive bias':partial_inductive,
             # 'factorized pos':partial_pos,
             'factorized pos':partial_pos2,
             }
        )
        return readable

