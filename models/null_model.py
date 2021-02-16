import math


class Null_Model:

    def __init__(self, tokens_count, tokens_total, alpha):

        self.alpha = alpha
        self.tokens_count = tokens_count
        self.tokens_total = tokens_total
        self.tokens_rank = {t: i + 1 for i, t in enumerate(sorted(self.tokens_count, key=lambda x: self.tokens_count[x], reverse=True))}

        self.null_logp = 0
        for token_label in self.tokens_count:
            token_logp = math.log(self.tokens_count[token_label] + self.alpha) - math.log(self.tokens_total)
            logp = self.rank_logp(token_label)
            p = math.exp(logp+token_logp)
            self.null_logp += p
        self.null_logp = math.log(self.null_logp)

    def rank_logp(self, token_label):
        if token_label in self.tokens_rank:
            rank = self.tokens_rank[token_label]
        else:
            rank = len(self.tokens_rank)
        p = 1 / (rank)
        logp = 0.5 * math.log(p)
        return logp

    def logp(self, amr, token_label, token_idx):
        logp = max(self.rank_logp(token_label), math.log(0.01))

        # punctuation
        if not token_label[0].isalpha() and not token_label[0].isdigit():
            return math.log(0.5)
        # coreference
        for spans in amr.coref:
            if any(token_idx in span for span in spans[1:]):
                return math.log(0.5)
        # parentheses
        if '(' in amr.tokens and ')' in amr.tokens:
            start_parens = [i for i in range(len(amr.tokens)) if amr.tokens[i]=='(']
            end_parens = [i for i in range(len(amr.tokens)) if amr.tokens[i]==')']
            for start, end in zip(start_parens, end_parens):
                if start <= token_idx <= end:
                    return math.log(0.5)

        # repetition
        for span in amr.spans:
            if span[0] < token_idx and ' '.join(amr.lemmas[t] for t in span) == token_label:
                if math.log(0.1)>logp:
                    return math.log(0.1)
        return logp

    def smoothing(self):
        total = 0
        for tok, rank in self.tokens_rank.items():
            p = 1 / math.sqrt(rank)
            total += p*self.tokens_count[tok]
        return total

    def inductive_bias(self, amr, token_label, token_idx):
        token_logp = math.log(self.tokens_count[token_label] + self.alpha) - math.log(self.tokens_total)
        joint_logp = self.logp(amr, token_label, token_idx) + token_logp
        logp = joint_logp - self.null_logp
        if logp>0:
            raise Exception('Improper Log Probability:',logp)
        return logp
