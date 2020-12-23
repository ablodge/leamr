import math
from collections import Counter
from statistics import stdev, mean

from amr_utils.alignments import AMR_Alignment

from models.base_model import Alignment_Model
from models.distance_model import Gaussian_Distance_Model, Skellam_Distance_Model
from models.null_model import Null_Model
from rule_based.subgraph_rules import fuzzy_align_subgraphs, postprocess_subgraph, clean_subgraph

ENGLISH = True

PARTIAL_CREDIT_RATE = 0.1


class Subgraph_Model(Alignment_Model):

    def __init__(self, amrs, alpha=0.1, partial_credit_alpha=0.01, align_duplicates=True):
        # Do not smooth over subgraphs (since the vocabulary is not finite or fixed)
        # The model uses backoff to a partial credit model instead
        super().__init__(amrs, alpha=alpha, smooth_translation=False)
        self.align_duplicates = align_duplicates

        self.distance_model = Skellam_Distance_Model()
        self.null_model = Null_Model(self)
        self.partial_credit_model = Partial_Credit_Subgraph_Model(self.tokens_count, self.tokens_total, alpha=partial_credit_alpha)
        self.partial_credit_model.update_parameters(amrs)

        self.is_initialized = False
        self.num_null_aligned = 0


    def trans_logp(self, amr, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        subgraph_label = self.get_alignment_label(amr, align)

        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        if not align.nodes:
            # null alignment
            trans_logp = self.null_model.logp(amr, token_label, align.tokens[0])
            if not self.is_initialized:
                trans_logp += math.log(PARTIAL_CREDIT_RATE)
            return trans_logp
        elif (token_label, subgraph_label) in self._trans_logp_memo:
            # check for memoized answer
            return self._trans_logp_memo[(token_label, subgraph_label)]
        elif token_label in self.translation_count and self.translation_count[token_label][subgraph_label]>0:
            # attested alignment
            trans_logp = math.log(self.translation_count[token_label][subgraph_label]) - math.log(self.translation_total)
            trans_logp -= token_logp
        elif len(token_label.split())>1 and any(t in self.translation_count and subgraph_label in self.translation_count[t]
                                                for t in token_label.split()):
            # partial match by token
            max_logp = float('-inf')
            for tok in token_label.split():
                if tok not in self.translation_count: continue
                if self.translation_count[tok][subgraph_label] == 0: continue
                logp = math.log(self.translation_count[tok][subgraph_label]) - math.log(self.translation_total)
                logp -= token_logp
                if logp > max_logp:
                    max_logp = logp
            trans_logp = max_logp + math.log(PARTIAL_CREDIT_RATE)
        else:
            # partial match by subgraph parts
            trans_logp = self.partial_credit_model.logp(amr, align, subgraph_label) + math.log(PARTIAL_CREDIT_RATE)

        self._trans_logp_memo[(token_label, subgraph_label)] = trans_logp

        return trans_logp

    def inductive_bias(self, amr, align):

        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        subgraph_label = self.get_alignment_label(amr, align)
        if not align.nodes:
            inductive_bias = self.null_model.inductive_bias(token_label)
        else:
            inductive_bias = self.partial_credit_model.inductive_bias(amr, align, subgraph_label)
        return inductive_bias


    def logp(self, amr, alignments, align):
        postprocess_subgraph(amr, alignments, align, english=ENGLISH)
        align = clean_subgraph(amr, alignments, align, english=ENGLISH)
        if align is None: return float('-inf')

        trans_logp = self.trans_logp(amr, align)
        dist_logp = self.distance_logp(amr, alignments, align)
        # inductive_bias = self.inductive_bias(amr, alignments, align)
        # ln( P(subgraph|tokens)*P(tokens|subgraph)*P(distance) )
        return trans_logp + dist_logp #+ inductive_bias

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
            dist = min(parent_dists)
            logp += self.distance_model.logp(dist)
            n+=1
        for dist in child_dists:
            logp += self.distance_model.logp(dist)
            n+=1
        if n < 1:
            return 0
        return logp/n

    def get_initial_alignments(self, amrs, preprocess=True):

        alignments = {}
        for j, amr in enumerate(amrs):
            print(f'\r{j} / {len(amrs)} preprocessed', end='')
            alignments[amr.id] = []
            for span in amr.spans:
                alignments[amr.id].append(AMR_Alignment(type='subgraph', tokens=span, amr=amr))
            if preprocess:
                fuzzy_align_subgraphs(amr, alignments, english=ENGLISH)
                for align in alignments[amr.id]:
                    postprocess_subgraph(amr, alignments, align, english=ENGLISH)
                    test = clean_subgraph(amr, alignments, align, english=ENGLISH)
                    if test is None:
                        align.nodes.clear()
        print('\r', end='')
        return alignments

    def update_parameters(self, amrs, alignments):
        super().update_parameters(amrs, alignments)

        # prune rare alignments
        for token_label in self.translation_count:
            for subgraph_label in list(self.translation_count[token_label].keys()):
                if self.tokens_count[token_label] == 0:
                    continue
                if self.translation_count[token_label][subgraph_label]/self.tokens_count[token_label]<=0.01:
                    # self.translation_count[token_label]['<null>'] += self.translation_count[token_label][subgraph_label]
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
        postprocess_subgraph(amr, alignments, tmp_align)
        candidate_neighbors = [s for s, r, t in amr.edges if t in tmp_align.nodes and s not in unaligned] + \
                              [t for s, r, t in amr.edges if s in tmp_align.nodes and t not in unaligned]
        for n2 in candidate_neighbors[:]:
            nalign = amr.get_alignment(alignments, node_id=n2)
            if not nalign:
                candidate_neighbors.remove(n2)
            if nalign.type == 'dupl-subgraph':
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
        if amr.nodes[n] == 'multi-sentence':
            candidate_spans2 = [span for span in candidate_spans if len(span) == 1]
            candidate_spans2 = [span for span in candidate_spans2 if
                                len(amr.tokens[span[0]]) == 1 and not amr.tokens[span[0]].isalpha() and not amr.tokens[span[0]].isdigit()]
            candidate_spans2 = [span for span in candidate_spans2 if span[-1]<len(amr.tokens)-1]
            if candidate_spans2:
                candidate_spans = candidate_spans2
        if amr.nodes[n] == 'person':
            candidate_spans = [span for span in candidate_spans if ' '.join(amr.lemmas[t] for t in span)=='person']
        if amr.nodes[n] == 'thing':
            candidate_spans = [span for span in candidate_spans if ' '.join(amr.lemmas[t] for t in span)=='thing']
        if amr.nodes[n] in ['multi-sentence', 'and', 'or'] and candidate_spans:
            candidate_neighbors = []
        for n2 in candidate_neighbors[:]:
            if amr.nodes[n2] in ['multi-sentence', 'and', 'or'] and candidate_spans:
                candidate_neighbors.remove(n2)
        if len([n2 for n2 in amr.nodes if amr.nodes[n]==amr.nodes[n2]])>1:
            for s, r, t in amr.edges:
                if t==n and amr.nodes[s] in ['include-91', 'same-01', 'instead-of-91', 'resemble-01', 'and', 'or']:
                    for s2, r2, t2 in amr.edges:
                        if s2 == s and t2 != t and amr.nodes[t2]==amr.nodes[n] and r2.endswith('1'):
                            candidate_spans = []
                            break
                        elif t2 == s and amr.nodes[s2]==amr.nodes[n] and r2.endswith('1-of'):
                            candidate_spans = []
                            break

        candidate_duplicates = []
        for n2 in amr.nodes:
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
            scores1[i] += self.inductive_bias(amr, new_align) - self.inductive_bias(amr, replaced_align)
            aligns1[i] = new_align
            # readable.append(self.readable_logp(amr,alignments, new_align))
        scores2 = {}
        aligns2 = {}
        for i, neighbor in enumerate(candidate_neighbors):
            replaced_align = amr.get_alignment(alignments, node_id=neighbor)
            new_align = AMR_Alignment(type=replaced_align.type, tokens=replaced_align.tokens, nodes=replaced_align.nodes+[n], amr=amr)
            scores2[i] = self.logp(amr, alignments, new_align) - self.logp(amr, alignments, replaced_align)
            scores2[i] += self.inductive_bias(amr, new_align) - self.inductive_bias(amr, replaced_align)
            aligns2[i] = new_align
            # readable.append(self.readable_logp(amr, alignments, new_align))
        scores3 = {}
        aligns3 = {}
        if self.align_duplicates:
            for i, span in enumerate(candidate_duplicates):
                new_align = AMR_Alignment(type='dupl-subgraph', tokens=span, nodes=[n], amr=amr)
                replaced_align = amr.get_alignment(alignments, token_id=span[0])
                scores3[i] = math.log(PARTIAL_CREDIT_RATE) + self.logp(amr, alignments, new_align) - self.logp(amr, alignments, replaced_align)
                scores3[i] += self.inductive_bias(amr, new_align) - self.inductive_bias(amr, replaced_align)
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

    def get_unaligned(self, amr, alignments):
        aligned = set()
        for align in alignments[amr.id]:
            aligned.update(align.nodes)
        return [n for n in amr.nodes if n not in aligned]

    def readable_logp(self, amr, alignments, align):
        readable = super().readable_logp(amr, alignments, align)
        subgraph_label = self.get_alignment_label(amr, align)

        trans_logp = self.trans_logp(amr, align)
        inductive_bias = self.inductive_bias(amr, align)
        if not align.nodes:
            partial_inductive = {'<null>':inductive_bias}
        else:
            partial_inductive = self.partial_credit_model.inductive_bias_readable(amr, align)
        dist_logp = self.distance_logp(amr, alignments, align)
        partial_scores = self.partial_credit_model.partial_logp(amr, align)
        readable['score'] += inductive_bias
        readable.update(
            {'subgraph':subgraph_label,
             'logP(subgraph|tokens)':trans_logp,
             'inductive bias': inductive_bias,
             'logP(distance)':dist_logp,
             'partial logp':partial_scores,
             'partial inductive bias':partial_inductive,
             }
        )
        return readable


class Partial_Credit_Subgraph_Model:

    def __init__(self, tokens_count, tokens_total, alpha):

        self.alpha = alpha

        self.concept_translation_count = {}
        self.concept_translation_total = 0
        self.concept_count = Counter()
        self.concept_total = 0

        self.edge_translation_count = {}
        self.edge_translation_total = 0
        self.edge_count = Counter()
        self.edge_total = 0

        self.tokens_count = tokens_count
        self.tokens_total = tokens_total

        self._trans_logp_memo = {}
        self._inductive_bias = {}

    def update_parameters(self, amrs):

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
        self.concept_translation_total += self.alpha*sum(len(self.concept_translation_count[t])+1 for t in self.concept_translation_count)
        self.edge_translation_total = \
            sum(self.edge_translation_count[t][p] for t in self.edge_translation_count for p in self.edge_translation_count[t])
        self.edge_translation_total += self.alpha*sum(len(self.edge_translation_count[t])+1 for t in self.edge_translation_count)

    def logp(self, amr, align, align_label):
        # ln( PARTIAL_CREDIT_RATE*P(root|tokens)*P(edge_1|tokens)*P(edge_2|tokens)... )
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        if (token_label, align_label) in self._trans_logp_memo:
            return self._trans_logp_memo[(token_label, align_label)]

        partial_logp = self.partial_logp(amr, align)
        logp = sum(partial_logp.values())
        logp = logp #+ math.log(PARTIAL_CREDIT_RATE)
        self._trans_logp_memo[(token_label, align_label)] = logp
        return logp

    def inductive_bias(self, amr, align, align_label):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        if token_label not in self.concept_translation_count:
            self.concept_translation_count[token_label] = Counter()
        if token_label not in self.edge_translation_count:
            self.edge_translation_count[token_label] = Counter()
        if (token_label, align_label) in self._inductive_bias:
            return self._inductive_bias[(token_label, align_label)]

        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)

        logp = 0
        concept_labels, _, edge_labels, source_labels = self.get_partial_alignment_labels(amr, align.nodes)
        for label in concept_labels:
            concept_logp = math.log(self.concept_count[label]+self.alpha) - math.log(self.concept_total)
            joint_logp = math.log(self.concept_translation_count[token_label][label] + self.alpha) \
                         - math.log(self.concept_translation_total)
            logp += joint_logp - concept_logp - token_logp
        for edge, source in zip(edge_labels, source_labels):
            edge_logp = math.log(self.edge_count[edge] + self.alpha) - math.log(self.edge_total)

            joint_logp = math.log(self.edge_translation_count[token_label][edge] + self.alpha) \
                     - math.log(self.edge_translation_total)
            logp += joint_logp - edge_logp - token_logp
        logp /= len(concept_labels+edge_labels)
        inductive_bias = logp
        self._inductive_bias[(token_label, align_label)] = inductive_bias
        return inductive_bias

    def inductive_bias_readable(self, amr, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        token_count = self.tokens_count[token_label]

        count = {f'Count(token={token_label})':token_count}
        concept_labels, _, edge_labels, source_labels = self.get_partial_alignment_labels(amr, align.nodes)
        for label in concept_labels:
            concept_count = self.concept_count[label]
            joint_count = self.concept_translation_count[token_label][label]
            count[f'Count({token_label},{label})'] = joint_count
            count[f'Count({label})'] = concept_count

        for edge, source in zip(edge_labels, source_labels):
            edge_count = self.edge_count[edge]
            joint_count = self.edge_translation_count[token_label][edge]
            count[f'Count({token_label},{edge})'] = joint_count
            count[f'Count({edge})'] = edge_count
        return count

    def concept_logp(self, token_label, concept_label):
        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
        if token_label not in self.concept_translation_count:
            self.concept_translation_count[token_label] = Counter()
        joint_logp = math.log(self.concept_translation_count[token_label][concept_label] + self.alpha) \
                         - math.log(self.concept_translation_total)
        logp = joint_logp - token_logp
        return logp

    def edge_conditional_logp(self, token_label, edge_label, source_label):
        token_logp = math.log(self.tokens_count[token_label] + self.alpha) - math.log(self.tokens_total)
        if token_label not in self.edge_translation_count:
            self.edge_translation_count[token_label] = Counter()
        joint_logp = math.log(self.edge_translation_count[token_label][edge_label] + self.alpha) \
                     - math.log(self.edge_translation_total)
        source_logp = self.concept_logp(token_label, source_label)
        logp = joint_logp - token_logp - source_logp
        return logp

    def get_partial_alignment_labels(self, amr, nodes):
        if not nodes:
            return [], [], [], []
        if len(nodes)==1:
            concept = amr.nodes[nodes[0]]
            concept = concept.replace(' ','_')
            return [concept], [concept], [], []
        edges = [(s, r, t) for s, r, t in amr.edges if s in nodes and t in nodes]
        roots = [n for n in nodes if not any(n==t for s,r,t in edges)]
        root_labels = [amr.nodes[n] for n in roots]
        root_labels = [s.replace(' ','_') for s in root_labels]
        concept_labels = [amr.nodes[n] for n in nodes]
        concept_labels = [s.replace(' ', '_') for s in concept_labels]
        edge_labels = [f'({amr.nodes[s]},{r},{amr.nodes[t]})' for s, r, t in edges]
        edge_labels = [s.replace(' ', '_') for s in edge_labels]
        source_labels = [amr.nodes[s].replace('"','') for s, r, t in edges]
        return concept_labels, root_labels, edge_labels, source_labels

    def partial_logp(self, amr, align):
        # joint probability of subgraph parts ln( P(n1)*P(edge=(n1,r1,n2))*P(edge=(n1,r2,n3))... )
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        _, root_labels, edge_labels, source_labels = self.get_partial_alignment_labels(amr, align.nodes)
        parts = {}
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