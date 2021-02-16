import math
from collections import Counter, defaultdict

from numpy import random

from rule_based.relation_rules import normalize_relation


def sample_random_noise():
    noise = random.beta(0.5, 0.5)
    return math.log(noise)


class Node_Model:

    def __init__(self, amrs, alpha=0.01):

        self.alpha = alpha
        self.translation_count = {}
        self.translation_total = 0
        self.concept_count = Counter()
        self.concept_total = 0
        self.tokens_count = Counter()
        self.tokens_total = 0
        self.add_noise = False
        self.first_iter = True
        self.init_params(amrs)

    def init_params(self, amrs):
        self.tokens_count = defaultdict(lambda: 0.)
        self.concept_count = defaultdict(lambda: 0.)
        for amr in amrs:
            tokens = {self.tokens_label(amr, span) for span in amr.spans}
            nodes = {self.concept_label(amr, n) for n in amr.nodes}
            for tok in tokens:
                self.tokens_count[tok] += 1/len(tokens)
                if tok not in self.translation_count:
                    self.translation_count[tok] = defaultdict(lambda: 0.)
                for node in nodes:
                    self.translation_count[tok][node] += 1/len(tokens)
            for node in nodes:
                self.concept_count[node] += 1/len(tokens)
        self.translation_total = len(amrs)
        self.concept_total = len(amrs)
        self.tokens_total = len(amrs)

    def concept_label(self, amr, n):
        return amr.nodes[n].replace(' ', '_')

    def tokens_label(self, amr, span):
        return ' '.join(amr.lemmas[t] for t in span)

    def update_parameters(self, amrs, alignments):

        self.translation_count = {}
        self.translation_total = 0
        self.concept_count = Counter()
        self.concept_total = 0
        self.tokens_count = Counter()
        self.tokens_total = 0

        for amr in amrs:
            for span in amr.spans:
                token_label = self.tokens_label(amr, span)
                self.tokens_count[token_label] += 1
            if amr.id not in alignments:
                continue
            for align in alignments[amr.id]:
                if align.type.startswith('dupl'): continue
                tokens = self.tokens_label(amr, align.tokens)
                if tokens not in self.translation_count:
                    self.translation_count[tokens] = Counter()
                nodes = {self.concept_label(amr, n) for n in align.nodes}
                for n_label in nodes:
                    self.translation_count[tokens][n_label] += 1
                    self.concept_count[n_label] += 1
        self.translation_total = sum(self.translation_count[t][s] for t in self.translation_count for s in self.translation_count[t])
        self.translation_total += self.alpha * sum(len(self.translation_count[t]) + 1 for t in self.translation_count)
        self.concept_total = self.translation_total
        self.tokens_total = self.translation_total
        self.add_noise = False
        self.first_iter = False

    def concept_logp(self, concept_label):
        alpha = 0.0 if self.first_iter else self.alpha
        logp = math.log(self.concept_count[concept_label] + alpha) \
                     - math.log(self.concept_total)
        return logp

    def concept_token_logp(self, concept_label, token_label):
        if token_label not in self.translation_count:
            return math.log(self.alpha) - math.log(self.translation_total)
        alpha = 0.0 if self.first_iter else self.alpha
        logp = math.log(self.translation_count[token_label][concept_label] + alpha) \
                     - math.log(self.translation_total)
        return logp

    def factorized_logp(self, amr, align):
        if not align: return {}
        token_label = self.tokens_label(amr, align.tokens)
        alpha = 0.0 if self.first_iter else self.alpha
        token_logp = math.log(self.tokens_count[token_label] + alpha) - math.log(self.tokens_total)
        node_logps = {}
        for n in align.nodes:
            n_label = self.concept_label(amr, n)
            joint_logp = self.concept_token_logp(n_label, token_label)
            node_logp = joint_logp - token_logp
            l = n_label
            i = 1
            while l in node_logps:
                i += 1
                l = f'{n_label}#{i}'
            l = f'{token_label} : {l}'
            node_logps[l] = node_logp
            if node_logps[l]>0:
                raise Exception('Improper Probability', token_label, l,
                                f'{self.translation_count[token_label][n_label]}/{self.tokens_count[token_label]}', node_logps[l])
        return node_logps

    def inductive_bias(self, amr, align):
        if not align: return {}
        token_label = self.tokens_label(amr, align.tokens)
        node_logps = {}
        for n in align.nodes:
            n_label = self.concept_label(amr, n)
            node_logp = self.concept_token_logp(n_label, token_label) - self.concept_logp(n_label)
            l = n_label
            i = 1
            while l in node_logps:
                i += 1
                l = f'{n_label}#{i}'
            l = f'{l} : {token_label}'
            node_logps[l] = node_logp
            if node_logps[l]>0:
                raise Exception('Improper Probability', token_label, l,
                                f'{self.translation_count[token_label][n_label]}/{self.concept_count[n_label]}', node_logps[l])
        return node_logps


class Internal_Edge_Model:

    def __init__(self, amrs, alpha=0.01, token_label_f=None):

        self.alpha = alpha
        self.translation_count = {}
        self.translation_total = 0
        self.node_translation_count = {}
        self.node_translation_total = 0
        self.edge_count = Counter()
        self.edge_total = 0
        self.tokens_count = Counter()
        self.tokens_total = 0
        self.add_noise = False
        self.first_iter = True
        self.token_label_f = token_label_f
        self.init_params(amrs)

    def init_params(self, amrs):
        self.tokens_count = defaultdict(lambda: 0.)
        self.concept_count = defaultdict(lambda: 0.)
        for amr in amrs:
            tokens = {self.tokens_label(amr, span) for span in amr.spans}
            edges = {(self.edge_label(amr, e), self.node_pair_label(amr, e)) for e in amr.edges}
            for tok in tokens:
                self.tokens_count[tok] += 1/len(tokens)
                if tok not in self.translation_count:
                    self.translation_count[tok] = defaultdict(lambda: 0.)
                    self.node_translation_count[tok] = defaultdict(lambda: 0.)
                for edge, node_pair in edges:
                    self.translation_count[tok][edge] += 1/len(tokens)
                    self.node_translation_count[tok][node_pair] += 1/len(tokens)
            for edge, node_pair in edges:
                self.edge_count[edge] += 1/len(tokens)
        self.translation_total = len(amrs)
        self.node_translation_total = len(amrs)
        self.edge_total = len(amrs)
        self.tokens_total = len(amrs)

    def edge_label(self, amr, e):
        s,r,t = e
        return f'({amr.nodes[s]},{r},{amr.nodes[t]})'.replace(' ', '_')

    def tokens_label(self, amr, span):
        if self.token_label_f is not None:
            return self.token_label_f(amr, span)
        return ' '.join(amr.lemmas[t] for t in span)

    def node_pair_label(self, amr, e):
        s,r,t = e
        return f'({amr.nodes[s]},{amr.nodes[t]})'.replace(' ', '_')

    def update_parameters(self, amrs, alignments):

        self.translation_count = {}
        self.translation_total = 0
        self.node_translation_count = {}
        self.node_translation_total = 0
        self.edge_count = Counter()
        self.edge_total = 0
        self.tokens_count = Counter()
        self.tokens_total = 0

        for amr in amrs:
            for span in amr.spans:
                token_label = self.tokens_label(amr, span)
                self.tokens_count[token_label] += 1
            if amr.id not in alignments:
                continue
            for align in alignments[amr.id]:
                if align.type.startswith('dupl'): continue
                tokens = self.tokens_label(amr, align.tokens)
                if tokens not in self.translation_count:
                    self.translation_count[tokens] = Counter()
                    self.node_translation_count[tokens] = Counter()
                if len(align.nodes)<=1: continue
                for e in amr.edges:
                    s,r,t = e
                    if s in align.nodes and t in align.nodes:
                        edge_label = self.edge_label(amr, e)
                        node_pair = self.node_pair_label(amr, e)
                        self.translation_count[tokens][edge_label] += 1
                        self.node_translation_count[tokens][node_pair] += 1
                        self.edge_count[edge_label] += 1
        self.translation_total = sum(self.translation_count[t][s] for t in self.translation_count for s in self.translation_count[t])
        self.translation_total += self.alpha * sum(len(self.translation_count[t]) + 1 for t in self.translation_count)

        self.node_translation_total = self.translation_total
        self.edge_total = self.translation_total
        self.tokens_total = self.translation_total
        self.add_noise = False
        self.first_iter = False

    def edge_logp(self, edge_label):
        alpha = 0.0 if self.first_iter else self.alpha
        logp = math.log(self.edge_count[edge_label] + alpha) \
               - math.log(self.edge_total)
        return logp

    def edge_token_logp(self, edge_label, token_label):
        if token_label not in self.translation_count:
            return math.log(self.alpha) - math.log(self.translation_total)
        alpha = 0.0 if self.first_iter else self.alpha
        logp = math.log(self.translation_count[token_label][edge_label] + alpha) \
               - math.log(self.translation_total)
        return logp

    def node_pair_logp(self, node_pair_label, token_label):
        if token_label not in self.node_translation_count:
            return math.log(self.alpha) - math.log(self.node_translation_total)
        alpha = 0.0 if self.first_iter else self.alpha
        logp = math.log(self.node_translation_count[token_label][node_pair_label] + alpha) \
               - math.log(self.node_translation_total)
        return logp

    def factorized_logp(self, amr, align):
        if len(align.nodes)<=1:
            return {}
        token_label = self.tokens_label(amr, align.tokens)
        alpha = 0.0 if self.first_iter else self.alpha
        token_logp = math.log(self.tokens_count[token_label] + alpha) - math.log(self.tokens_total)
        edge_logps = {}
        edges = [(s,r,t) for s,r,t in amr.edges if s in align.nodes and t in align.nodes]
        for e in edges:
            e_label = self.edge_label(amr, e)
            s_t_label = self.node_pair_label(amr, e)
            s_t_logp = self.node_pair_logp(s_t_label, token_label) - token_logp
            edge_logp = self.edge_token_logp(e_label, token_label) - token_logp
            l = e_label
            i = 1
            while l in edge_logps:
                i += 1
                l = f'{e_label}#{i}'
            l = f'{token_label} : {l}'
            edge_logps[l] = edge_logp - s_t_logp
            if edge_logps[l]>0:
                raise Exception('Improper Probability', token_label, l,
                                f'{self.translation_count[token_label][e_label]}/{self.tokens_count[token_label]}', edge_logps[l])
        return edge_logps

    def inductive_bias(self, amr, align):
        if len(align.nodes)<=1:
            return {}
        token_label = self.tokens_label(amr, align.tokens)
        edge_logps = {}
        edges = [(s, r, t) for s, r, t in amr.edges if s in align.nodes and t in align.nodes]
        for s, r, t in edges:
            e_label = self.edge_label(amr, (s, r, t))
            edge_logp = self.edge_token_logp(e_label, token_label) - self.edge_logp(e_label)
            l = e_label
            i = 1
            while l in edge_logps:
                i += 1
                l = f'{e_label}#{i}'
            l = f'{l} : {token_label}'
            edge_logps[l] = edge_logp
            if edge_logps[l]>0:
                raise Exception('Improper Probability', token_label, l,
                                f'{self.translation_count[token_label][e_label]}/{self.edge_count[e_label]}', edge_logps[l])
        return edge_logps


class External_Edge_Model(Internal_Edge_Model):

    def __init__(self, amrs, alpha=0.01):
        super().__init__(amrs, alpha)

    def edge_label(self, amr, e):
        s,r,t = normalize_relation(e)
        return r

    def factorized_logp(self, amr, align):
        token_label = self.tokens_label(amr, align.tokens)
        token_logp = math.log(self.tokens_count[token_label] + self.alpha) - math.log(self.tokens_total)
        edge_logps = {}
        edges = [(s, r, t) for s, r, t in amr.edges if s in align.nodes and t in align.nodes]
        for s, r, t in edges:
            e_label = self.edge_label(amr, (s, r, t))
            edge_logp = self.edge_token_logp(e_label, token_label) - token_logp
            l = e_label
            i = 0
            while l in edge_logps:
                i += 1
                l = f'{e_label}#{i}'
            l = f'{token_label} : {l}'
            edge_logps[l] = edge_logp
        return edge_logps


class POS_Node_Model(Node_Model):

    def __init__(self, amrs, alpha):
        self.null_count = Counter()
        self.null_total = 0
        super().__init__(amrs, alpha)

    def tokens_label(self, amr, span):
        return amr.pos[span[0]]

    def init_params(self, amrs):
        self.tokens_count = defaultdict(lambda: 0.)
        self.concept_count = defaultdict(lambda: 0.)
        for amr in amrs:
            tokens = [self.tokens_label(amr, span) for span in amr.spans]
            nodes = [self.concept_label(amr, n) for n in amr.nodes]
            for tok in tokens:
                self.tokens_count[tok] += 1
            for node in nodes:
                self.concept_count[node] += 1
        self.concept_total = sum(self.concept_count.values())
        self.tokens_total = sum(self.tokens_count.values())
        for tok in self.tokens_count:
            self.translation_count[tok] = defaultdict(lambda:0)
            for concept in self.concept_count:
                self.translation_count[tok][concept] = self.concept_count[concept]/len(self.tokens_count)
        self.translation_total = self.concept_total
        for t in self.tokens_count:
            self.null_count[t] = 1
        self.null_total = len(self.tokens_count)

    def update_parameters(self, amrs, alignments):
        super().update_parameters(amrs, alignments)
        self.null_count = Counter()
        self.null_total = 0
        for amr in amrs:
            for align in alignments[amr.id]:
                if not align.nodes:
                    self.null_total += 1
                    pos = self.tokens_label(amr, align.tokens)
                    self.null_count[pos] += 1
        self.null_total += self.alpha*(len(self.tokens_count)+1)

    def inductive_bias(self, amr, align):
        parts = super().inductive_bias(amr, align)
        if not align.nodes:
            token_label = self.tokens_label(amr, align.tokens)
            alpha = 0 if self.first_iter else self.alpha
            parts[f'<null> : {token_label}'] = math.log(self.null_count[token_label] + alpha) - math.log(self.null_total)
        return parts

