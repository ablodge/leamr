import math

from models.base_model import Alignment_Model


class Null_Model:

    def __init__(self, align_model:Alignment_Model):

        self.alpha = align_model.alpha
        self.tokens_count = align_model.tokens_count
        self.tokens_total = align_model.tokens_total
        self.tokens_rank = {t: i + 1 for i, t in enumerate(sorted(self.tokens_count, key=lambda x: self.tokens_count[x], reverse=True))}

    def logp(self, token_label):
        if token_label in self.tokens_rank:
            rank = self.tokens_rank[token_label]
        else:
            rank = len(self.tokens_rank)
        p = 1 / (rank)
        logp = 0.5 * math.log(p)
        logp = max(logp, math.log(0.05))
        return logp

    def inductive_bias(self, token_label):
        return 0.0
        # return math.log(self.tokens_count[token_label]+self.alpha) - math.log(self.tokens_total)
