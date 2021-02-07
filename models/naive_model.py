import math
from collections import Counter

from rule_based.relation_rules import normalize_relation


class Concept_Edge_Model:

    def __init__(self, mode='subgraph'):

        self.alpha = 0.01
        self.mode = mode
        if mode not in ['subgraph','relation']:
            raise Exception('Unknown Mode:', mode)

        # percent of AMRs with a given token_label and concept_label
        self.concept_translation_count = {}
        # percent of AMRs with a given concept_label
        self.concept_count = Counter()
        # percent of AMRs with a given token_label and edge_label
        self.edge_translation_count = {}
        # percent of AMRs with a given edge_label
        self.edge_count = Counter()
        # percent of AMRs with a given token_label
        self.tokens_count = Counter()

        self.amrs_total = 0

        self._inductive_bias = {}


    def concept_label(self, amr, n):
        return amr.nodes[n].replace(' ', '_')

    def edge_label(self, amr, e):
        s,r,t = e
        if self.mode == 'subgraph':
             return f'({amr.nodes[s]},{r},{amr.nodes[t]})'.replace(' ', '_')
        else:
            s,r,t = normalize_relation(e)
            return r

    def update_parameters(self, amrs):

        for amr in amrs:
            # concept stats
            concepts = [self.concept_label(amr, n) for n in amr.nodes]
            edges = [self.edge_label(amr, e) for e in amr.edges]
            all_tokens = [' '.join(amr.lemmas[t] for t in span) for span in amr.spans]
            for token_label in set(all_tokens):
                self.tokens_count[token_label]+=1
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

        self.amrs_total = len(amrs)

    def inductive_bias(self, amr, align, align_label):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        if (token_label, align_label) in self._inductive_bias:
            return self._inductive_bias[(token_label, align_label)]

        pmis = self.inductive_bias_readable(amr, align)
        inductive_bias = sum(pmis.values())
        inductive_bias /= len(pmis)

        self._inductive_bias[(token_label, align_label)] = inductive_bias
        return inductive_bias

    def inductive_bias_readable(self, amr, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        if token_label not in self.concept_translation_count:
            self.concept_translation_count[token_label] = Counter()
        amrs_total = self.amrs_total + self.alpha*(self.amrs_total + 1)

        token_logp = math.log(self.tokens_count[token_label]+self.alpha) - math.log(amrs_total)

        readable = {}
        if self.mode=='subgraph':
            labels, edge_labels, _, _ = self.get_subgraph_labels(amr, align.nodes)
        else:
            labels = self.get_relation_labels(amr, align.edges)
        for label in labels:
            label_logp = math.log(self.concept_count[label] + self.alpha) - math.log(amrs_total)
            joint_logp = math.log(self.concept_translation_count[token_label][label] + self.alpha) \
                         - math.log(amrs_total)
            readable[f'{token_label}:{label}'] = joint_logp - label_logp - token_logp
        # for edge, source in zip(edge_labels, source_labels):
        #     edge_logp = math.log(self.edge_count[edge] + self.alpha) - math.log(self.edge_total)
        #
        #     joint_logp = math.log(self.edge_translation_count[token_label][edge] + self.alpha) \
        #                  - math.log(self.edge_translation_total)
        #     readable[f'{token_label}:{edge}'] = joint_logp - edge_logp - token_logp
        # logp /= len(concept_labels+edge_labels)
        return readable

    def concept_logp(self, token_label, concept_label):
        if token_label not in self.concept_translation_count:
            return 0.0
        token_count = self.tokens_count[token_label]
        logp = math.log(self.concept_translation_count[token_label][concept_label] + self.alpha) \
                     - math.log(token_count+self.alpha*(len(self.concept_translation_count[token_label])+1))
        return logp

    def edge_conditional_logp(self, token_label, edge_label, source_label=None):
        if token_label not in self.edge_translation_count:
            return 0.0
        if source_label:
            count = self.concept_translation_count[token_label][source_label]
        else:
            count = self.tokens_count[token_label]
        logp = math.log(self.edge_translation_count[token_label][edge_label] + self.alpha) \
                     - math.log(count+self.alpha*(len(self.edge_translation_count[token_label])+1))
        return logp

    def get_relation_labels(self, amr, edges):
        return [self.edge_label(amr, e) for e in edges]

    def get_subgraph_labels(self, amr, nodes):
        if not nodes:
            return [], [], [], []
        if len(nodes)==1:
            concept = amr.nodes[nodes[0]]
            concept = concept.replace(' ','_')
            return [concept], [concept], [], []
        edges = [(s, r, t) for s, r, t in amr.edges if s in nodes and t in nodes]
        concept_labels = [self.concept_label(amr, n) for n in nodes]
        edge_labels = [self.edge_label(amr, e) for e in edges]
        source_labels = [self.concept_label(amr, s).replace('"','') for s, r, t in edges]
        target_labels = [self.concept_label(amr, t).replace('"','') for s, r, t in edges]
        return concept_labels, edge_labels, source_labels, target_labels

    def factorized_logp(self, amr, align):
        # ln(P(root|tokens)*P(edge_1|tokens)*P(edge_2|tokens)... )
        partial_logp = self.factorized_logp_readable(amr, align)
        logp = sum(partial_logp.values())
        return logp

    def factorized_logp_readable(self, amr, align):
        # joint probability of subgraph parts ln( P(n1)*P(edge=(n1,r1,n2))*P(edge=(n1,r2,n3))... )
        if not align: return {}
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        parts = {}
        if self.mode=='relation':
            edge_labels = self.get_relation_labels(amr, align.edges)
            for edge in edge_labels:
                p = edge
                i = 0
                while p in parts:
                    i += 1
                    p = f'{edge}:{i}'
                edge_logp = self.edge_conditional_logp(token_label, edge)
                parts[p] = edge_logp
            return parts
        concept_labels, edge_labels, source_labels, target_labels = self.get_subgraph_labels(amr, align.nodes)
        for label in concept_labels:
            if label in target_labels: continue
            p = label
            i = 0
            while p in parts:
                i += 1
                p = f'{label}:{i}'
            parts[p] = self.concept_logp(token_label, label)
        if not parts:
            parts[concept_labels[0]] = self.concept_logp(token_label, concept_labels[0])
        for edge, source, target in zip(edge_labels, source_labels, target_labels):
            p = edge
            i = 0
            while p in parts:
                i += 1
                p = f'{edge}:{i}'
            edge_logp = self.edge_conditional_logp(token_label, edge, source)
            parts[p] = edge_logp
        return parts