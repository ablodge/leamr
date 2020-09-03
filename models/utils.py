import math
import sys

from tqdm import tqdm

# requires class that implements:
# - get_initial_alignments(amrs)
# - get_unaligned(amr, aignments)
# - get_scores(amr, alignments, n, unaligned)
# - logp(amr, alignments, align)
# - readable_logp(amr, alignments, align)

def align_all(model, amrs, alignments=None):
    if alignments is None:
        alignments = model.get_initial_alignments(amrs)

    perplexity = 0
    N = 0
    for amr in tqdm(amrs, file=sys.stdout):
        unaligned = model.get_unaligned(amr, alignments)

        while unaligned:
            all_scores = {}
            candidate_aligns = {}
            partitions = {}

            for n in unaligned:
                n_aligns, n_scores, n_partions = model.get_scores(amr, alignments, n, unaligned)
                for span in n_scores:
                    all_scores[(n,span)] = n_scores[span]
                    candidate_aligns[(n,span)] = n_aligns[span]
                    partitions[(n,span)] = n_partions[span]

            candidates = [(n, span) for n, span in all_scores if n in unaligned]
            new_scores = {}
            for n,span in candidates:
                new_scores[(n,span)] = all_scores[(n,span)]

            best = max(new_scores.keys(), key=lambda x: new_scores[x])
            n, span = best
            span = list(span)
            new_align = candidate_aligns[best]
            # readable = [(math.exp(new_scores[(n,span)]),
            #             amr.nodes[n],
            #              ' '.join(amr.lemmas[t] for t in span),
            #              model.readable_logp(amr, alignments, candidate_aligns[(n,span)]),
            #              ) for n,span in new_scores.keys()]
            # readable = [x for x in sorted(readable, key=lambda y :y[0], reverse=True)]
            # print()

            # add node to alignment
            for i,align in enumerate(alignments[amr.id]):
                if align.tokens == span and not align.type.startswith('dupl'):
                    alignments[amr.id][i] = new_align
                    break

            unaligned = model.get_unaligned(amr, alignments)

        amr.alignments = alignments[amr.id]
        for align in alignments[amr.id]:
            logp = model.logp(amr, alignments, align)
            p = math.exp(logp)
            ent = - p* math.log(p, 2) if p > 0 else 0
            perplexity += math.pow(2, ent)
            N += 1
    perplexity /= N
    print(f'Perplexity: {perplexity}')
    return alignments



