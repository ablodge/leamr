# LEAMR 

**LEAMR** (**L**inguistically **E**nriched **AMR**) Alignments
A release of models and data related to alignments between AMR and English text for better parsing and probing of many different linguistic phenomena.

For more details, read our paper.
Austin Blodgett and Nathan Schneider. 2021. Probabilistic, Structure-Aware Algorithms for Improved Variety, Accuracy, and Coverage of AMR Alignments. In Proceedings of the 59th Annual Meeting ofthe Association for Computational Linguistics.


Also see [AMR-utils](https://github.com/ablodge/amr-utils) and the [AMR Bibliography](https://nert-nlp.github.io/AMR-Bibliography/).




# Data

AMR Release 3.0 [https://catalog.ldc.upenn.edu/LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02)
Little Prince [https://amr.isi.edu/download.html](https://amr.isi.edu/download.html)
as well as 350 sentenes with gold alignments in leamr_test.txt and leamr_dev.txt.

We release 4 layers of alignments: **subgraph**, **duplicate subgraph**, **relation**, and **reentrancy** alignments. 

For AMR Release 3.0 and Little Prince, we release:

- `<corpus>.subgraph_alignments.json`: containing **subgraph** and **duplicate subgraph** alignments. alignments of connected DAG-shaped subgraphs to a single span.  includes alignments of connected DAG-shaped subgraphs to a span, where the semantic content is a duplicate of another subgraph (This is usually the result of some linguistic mechanism, such as coordination or ellipsis.)
- `<corpus>.relation_alignments.json`: **relation** alignments of relations to tokens cueing that relation, including argument structures and single relation alignments
- `<corpus>.reentrancy_alignments.json`: **reentrancy** alignments of reentrancy edges to their ``triggers'' for phenomona like coreference, control, and coordination. 


Also `<corpus>.spans.json`: spans for each sentence, grouping tokens which are either named entities or multiword expressions


## JSON Format
Alignments are released as JSON files.

To read alignments from a JSON file do:
```
reader = AMR_Reader()
alignments = reader.load_alignments_from_json(alignments_file)
```

# Get Alignments
Anonymized alignments are stored in the folder `data-release/alignments`. To get the AMR and 

```
pip install requirements.txt
wget https://amr.isi.edu/download/amr-bank-struct-v1.6.txt -O data-release/amrs/little_prince.txt
python build_data.py <LDC parent dir>
python unanonymize_alignments.py
```

# Run Pre-trained Aligner
```
python align_with_pretrained_model -T ldc+little_prince.txt -t <unaligned amr file>
```

# Train Aligner
```
python train_subgraph_aligner.py -T ldc+little_prince.txt
mv ldc+little_prince.subgraph_alignments.epoch4.json ldc+little_prince.subgraph_alignments.json 
python train_relation_aligner.py -T ldc+little_prince.txt
mv ldc+little_prince.relation_alignments.epoch4.json ldc+little_prince.relation_alignments.json 
python train_reentrancy_aligner.py -T ldc+little_prince.txt
mv ldc+little_prince.reentrancy_alignments.epoch4.json ldc+little_prince.reentrancy_alignments.json 
```

# Bibtex
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
