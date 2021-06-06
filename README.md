# LEAMR (Linguistically Enriched AMR) Alignments
A release of models and data related to alignments between AMR and English text for better parsing and probing of many different linguistic phenomena.

Also see the [AMR-utils](https://github.com/ablodge/amr-utils) library.

## Data (Planned to be Released)

- `<corpus>.spans.json`: spans for each sentence, grouping tokens which are either named entities or multiword expressions
- `<corpus>.subgraph_alignments.json`: alignments of connected DAG-shaped subgraphs to a single span
- `<corpus>.relation_alignments.json`: alignments of relations to tokens cueing that relation, including argument structures and single relation alignments
- `<corpus>.duplicate_alignments.json`: alignments of connected DAG-shaped subgraphs to a span, where the semantic content is a duplicate of another subgraph (This is usually the result of some linguistic mechanism, such as coordination or ellipsis.)
- `<corpus>.reentrancy_alignments.json`: alignments of reentrancy edges to their ``triggers'' for phenomona like coreference, control, and coordination. 


## Get Alignments
AMR Release 3.0 [https://catalog.ldc.upenn.edu/LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02)
Little Prince [https://amr.isi.edu/download.html](https://amr.isi.edu/download.html)

## Run Pre-trained Aligner


## Train Aligner
