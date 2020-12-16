import math

from models.base_model import Alignment_Model


class Null_Model:

    def __init__(self, align_model:Alignment_Model):

        self.alpha = align_model.alpha
        self.tokens_count = align_model.tokens_count
        self.tokens_total = align_model.tokens_total
        self.tokens_rank = {t: i + 1 for i, t in enumerate(sorted(self.tokens_count, key=lambda x: self.tokens_count[x], reverse=True))}

    def logp(self, amr, token_label, token_idx):
        if token_label in self.tokens_rank:
            rank = self.tokens_rank[token_label]
        else:
            rank = len(self.tokens_rank)
        # punctuation
        if not token_label[0].isalpha() and not token_label[0].isdigit():
            return math.log(0.5)
        # coreference
        for spans in amr.coref:
            if any(token_idx in span for span in spans[1:]):
                return math.log(0.5)
        # repetition
        for span in amr.spans:
            if span[0]!=token_idx and ' '.join(amr.lemmas[t] for t in span)==token_label and rank>=4:
                return math.log(0.5)
        p = 1 / (rank)
        logp = 0.5 * math.log(p)
        logp = max(logp, math.log(0.01))
        return logp

    def smoothing(self):
        total = 0
        for tok, rank in self.tokens_rank.items():
            p = 1 / math.sqrt(rank)
            total += p*self.tokens_count[tok]
        return total


    def inductive_bias(self, token_label):
        return 0.0
