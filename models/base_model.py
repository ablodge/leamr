import sys
from collections import Counter

from tqdm import tqdm


class Alignment_Model:

    def __init__(self, amrs, alpha=0.01, smooth_translation=False):

        self.alpha = alpha
        self.smooth_translation = smooth_translation
        self.translation_count = {}
        self.translation_total = 0

        self.tokens_count = Counter()
        self.tokens_total = 0
        for amr in amrs:
            for span in amr.spans:
                token_label = ' '.join(amr.lemmas[t] for t in span)
                self.tokens_count[token_label] += 1
        self.tokens_total = sum(self.tokens_count[t] for t in self.tokens_count)
        self.tokens_total += self.alpha*(len(self.tokens_count)+1)

        self._trans_logp_memo = {}

    def logp(self, amr, alignments, align):
        return 0

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

        for amr in amrs:
            if amr.id not in alignments:
                continue
            for align in alignments[amr.id]:
                tokens = ' '.join(amr.lemmas[t] for t in align.tokens)
                if tokens not in self.translation_count:
                    self.translation_count[tokens] = Counter()
                align_label = self.get_alignment_label(amr, align)
                if align_label is None:
                    continue
                self.translation_count[tokens][align_label] += 1

        self.translation_total = sum(self.translation_count[t][s] for t in self.translation_count for s in self.translation_count[t])
        if self.smooth_translation:
            self.translation_total += self.alpha*sum(len(self.translation_count[t])+1 for t in self.translation_count)

    def align(self, amr, alignments, n, unaligned=None, return_all=False):
        pass

    def get_unaligned(self, amr, alignments):
        return []

    def align_all(self, amrs, alignments=None, preprocess=True):
        if alignments is None:
            alignments = self.get_initial_alignments(amrs, preprocess)

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

                # old_alignments = {tuple(align.tokens): align for align in alignments[amr.id]}
                # readable = [(all_scores[(n,span)],
                #             self.get_alignment_label(amr, candidate_aligns[(n,span)]),
                #              ' '.join(amr.lemmas[t] for t in span),
                #              self.readable_logp(amr, alignments, candidate_aligns[(n,span)]),
                #              self.readable_logp(amr, alignments, old_alignments[span]),
                #              ) for n,span in all_scores.keys()]
                # readable = [x for x in sorted(readable, key=lambda y :y[0], reverse=True)]
                # x = 0

                # add node to alignment
                found = False
                for i, align in enumerate(alignments[amr.id]):
                    if align.tokens == span and align.type == new_align.type:
                        if align.type == 'dupl-subgraph':
                            continue
                        alignments[amr.id][i] = new_align
                        found = True
                        break
                if not found:
                    alignments[amr.id].append(new_align)

                unaligned = self.get_unaligned(amr, alignments)

            amr.alignments = alignments[amr.id]

        return alignments
