import math
import sys
from collections import Counter

from tqdm import tqdm


class Alignment_Model:

    def __init__(self, amrs, alpha=0.01):

        self.alpha = alpha
        self.translation_count = {}
        self.translation_total = 0

        self.tokens_count = Counter()
        self.tokens_total = 0
        for amr in amrs:
            for span in amr.spans:
                token_label = ' '.join(amr.lemmas[t] for t in span)
                self.tokens_count[token_label] += 1
        self.tokens_total = sum(self.tokens_count[t] + self.alpha for t in self.tokens_count)

        self._trans_logp_memo = {}

    def logp(self, amr, alignments, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        align_label = self.get_alignment_label(amr, align)
        token_logp = math.log(self.tokens_count[token_label] + self.alpha) - math.log(self.tokens_total)

        if (token_label, align_label) in self._trans_logp_memo:
            return self._trans_logp_memo[(token_label, align_label)]
        elif token_label in self.translation_count:
            trans_logp = math.log(self.translation_count[token_label][align_label] + self.alpha) - math.log(self.translation_total)
            trans_logp -= token_logp
        else:
            trans_logp = math.log(self.alpha) - math.log(self.translation_total)
            trans_logp -= token_logp

        self._trans_logp_memo[(token_label, align_label)] = trans_logp

        return trans_logp

    def readable_logp(self, amr, alignments, align):
        token_label = ' '.join(amr.lemmas[t] for t in align.tokens)
        tokens_count = self.tokens_count[token_label]
        score = self.logp(amr, alignments, align)
        return {'tokens':token_label,
                'score':score,
                'tokens_count': tokens_count,
                }

    def get_alignment_label(self, amr, ns):
        return str(ns)

    def get_initial_alignments(self, amrs, preprocess=True):
        return {amr.id:[] for amr in amrs}

    def update_parameters(self, amrs, alignments):
        self.translation_count = {}
        self.translation_total = 0
        self._trans_logp_memo = {}

        align_labels = set()
        for amr in amrs:
            if amr.id not in alignments:
                continue
            for align in alignments[amr.id]:
                tokens = ' '.join(amr.lemmas[t] for t in align.tokens)
                if tokens not in self.translation_count:
                    self.translation_count[tokens] = Counter()
                align_label = self.get_alignment_label(amr, align)
                align_labels.add(align_label)
                self.translation_count[tokens][align_label] += 1

        self.translation_total = sum(self.translation_count[t][s] for t in self.translation_count for s in self.translation_count[t])
        self.translation_total += self.alpha * len(self.tokens_count) * len(align_labels)

    def align(self, amr, alignments, n, unaligned=None, return_all=False):
        pass

    def get_unaligned(self, amr, alignments):
        return []

    def align_all(self, amrs, alignments=None, preprocess=True):
        if alignments is None:
            alignments = self.get_initial_alignments(amrs, preprocess)

        perplexity = 0
        N = 0
        for amr in tqdm(amrs, file=sys.stdout):
            unaligned = self.get_unaligned(amr, alignments)

            while unaligned:
                all_scores = {}
                candidate_aligns = {}

                for n in unaligned:
                    aligns, scores = self.align(amr, alignments, n, unaligned, return_all=True)
                    if aligns is None: continue
                    # span = tuple(best_align.tokens)
                    for span in aligns:
                        all_scores[(n, span)] = scores[span]
                        candidate_aligns[(n, span)] = aligns[span]
                if not all_scores:
                    break

                best = max(all_scores.keys(), key=lambda x: all_scores[x])
                n, span = best
                span = list(span)
                new_align = candidate_aligns[best]

                old_alignments = {tuple(align.tokens): align for align in alignments[amr.id]}
                readable = [(all_scores[(n,span)],
                            self.get_alignment_label(amr, candidate_aligns[(n,span)]),
                             ' '.join(amr.lemmas[t] for t in span),
                             self.readable_logp(amr, alignments, candidate_aligns[(n,span)]),
                             self.readable_logp(amr, alignments, old_alignments[span]),
                             ) for n,span in all_scores.keys()]
                readable = [x for x in sorted(readable, key=lambda y :y[0], reverse=True)]
                x = 0

                # add node to alignment
                for i, align in enumerate(alignments[amr.id]):
                    if align.tokens == span and align.type == new_align.type:
                        if align.type == 'dupl-subgraph' and any(n not in new_align.nodes for n in align.nodes):
                            continue
                        alignments[amr.id][i] = new_align
                        break

                unaligned = self.get_unaligned(amr, alignments)

            amr.alignments = alignments[amr.id]
            for align in alignments[amr.id]:
                logp = self.logp(amr, alignments, align)
                p = math.exp(logp)
                ent = - p * math.log(p, 2) if p > 0 else 0
                perplexity += math.pow(2, ent)
                N += 1
        perplexity /= N
        print(f'Perplexity: {perplexity}')
        return alignments