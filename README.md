# LEAMR 

**LEAMR** (**L**inguistically **E**nriched **AMR**) Alignments
A release of models and data related to alignments between AMR and English text for better parsing and probing of many different linguistic phenomena.

Also see [AMR-utils](https://github.com/ablodge/amr-utils) and the [AMR Bibliography](https://nert-nlp.github.io/AMR-Bibliography/).




# Data

4 types of alignments: **subgraph**, **duplicate subgraph**, **relation**, and **reentrancy** alignments:

- **subgraphs** `<corpus>.subgraph_alignments.json`: alignments of connected DAG-shaped subgraphs to a single span
    - **duplicate subgraphs** includes alignments of connected DAG-shaped subgraphs to a span, where the semantic content is a duplicate of another subgraph (This is usually the result of some linguistic mechanism, such as coordination or ellipsis.)
- **relations** `<corpus>.relation_alignments.json`: alignments of relations to tokens cueing that relation, including argument structures and single relation alignments
- **reentrancies** `<corpus>.reentrancy_alignments.json`: alignments of reentrancy edges to their ``triggers'' for phenomona like coreference, control, and coordination. 


Also `<corpus>.spans.json`: spans for each sentence, grouping tokens which are either named entities or multiword expressions


## JSON Format
Alignments are released as JSON files.

To read alignments from a JSON file do:
```
reader = AMR_Reader()
alignments = reader.load_alignments_from_json(alignments_file)
```

# Get Alignments
AMR Release 3.0 [https://catalog.ldc.upenn.edu/LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02)
Little Prince [https://amr.isi.edu/download.html](https://amr.isi.edu/download.html)


```
wget https://amr.isi.edu/download/amr-bank-struct-v1.6.txt -O data-release/amrs/little_prince.txt
python build_data.py <LDC parent dir>
```

# Run Pre-trained Aligner


# Train Aligner


# How to cite LEAMR
You can cite our paper "Probabilistic, Structure-Aware Algorithms for Improved Variety, Accuracy, and Coverage of AMR Alignments".

Bibtex:
```
@inproceedings{blodgett2021,
    title = "Probabilistic, Structure-Aware Algorithms for Improved Variety, Accuracy, and Coverage of {AMR} Alignments",
    author = "Blodgett, Austin and
      Schneider, Nathan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
    month = aug,
    year = 2021
}
```
