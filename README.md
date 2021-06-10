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


## Subgraph Alignments

span to subgraph, named entities, mwes
```
{"type": "subgraph", "tokens": [3], "nodes": ["1.1", "1.1.1"], "edges": [["1.1", ":ARG0-of", "1.1.1"]], "string": "students => person :ARG0-of study-01"}
```

```
{"type": "subgraph", "tokens": [7, 8], "nodes": ["1.2.2", "1.2.2.1", "1.2.2.1.1", "1.2.2.1.2"], "edges": [["1.2.2.1", ":name", "1.2.2.1.1"], ["1.2.2.1", ":op1", "1.2.2.1.2"], ["1.2.2", ":op2", "1.2.2.1"]], "string": "New York => city :name name, name :op1 New, name :op2 York"}
```

## Duplicate Subgraph Alignments

ellipsis

```
{"type": "dupl-subgraph", "tokens": [3], "nodes": ["1.1.2.1","1.1.2.1.1"]}, "edges": [["1.1.2.1",":ARG0-of","1.1.2.1.1"]], "string": "students => person :ARG0-of study-01"}
```

## Relation Alignments

arg structures

```
{"type": "relation", "tokens": [4], "nodes": [], "edges": [["1", ":ARG0", "1.1"], ["1", ":ARG1", "1.2"]], "string": "want => want-01 :ARG0 person, want-01 :ARG1 visit-01"}
```

single rel

```
{"type": "relation", "tokens": [9], "nodes": [], "edges": [["1.2", ":time", "1.2.3"]], "string": "when => visit-01 :time graduate-01"}
```

## Reentrancy Alignments and Classes

examples coref, coordination, control

```
{"type": "reentrancy:coref", "tokens": [10], "edges": [["1.2.3", ":ARG0", "1.1"]], "string": "they => graduate-01 :ARG0 person"}
{"type": "reentrancy:control", "tokens": [4], "edges": [["1.2", ":ARG0", "1.1"]], "string": "want => visit-01 :ARG0 person"}
```

## JSON Format
The package includes tools for converting AMR alignments from and to JSON like the following.
```
....
```

The advantages of using JSON are:
- Easy to load and save (No need to write a special script for reading some esoteric format)
- Can store additional information in a `type` to distinguish different types of alignments
- Can easily store multiple sets of alignments separately, without needing to modify an AMR file. That makes it easy to compare different sets of alignments. 

To read alignments from a JSON file do:
```
reader = AMR_Reader()
alignments = reader.load_alignments_from_json(alignments_file)
```
To save alignments to a JSON file do:
```
reader = AMR_Reader()
reader.save_alignments_to_json(alignments_file, alignments)
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
