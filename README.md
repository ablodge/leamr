# LEAMR 

**LEAMR** (**L**inguistically **E**nriched **AMR**, pronounced _lemur_) Alignments is a data release of alignments between AMR and English text for better parsing and probing of many different linguistic phenomena. We also include our code for the LEAMR aligner. For more details, read our paper.

Austin Blodgett and Nathan Schneider. 2021. _Probabilistic, Structure-Aware Algorithms for Improved Variety, Accuracy, and Coverage of AMR Alignments_. In Proceedings of the 59th Annual Meeting ofthe Association for Computational Linguistics.

For other useful resouces for AMR research, also take a look at [AMR-utils](https://github.com/ablodge/amr-utils) and the [AMR Bibliography](https://nert-nlp.github.io/AMR-Bibliography/).


# Install

```
pip install requirements.txt
git clone https://github.com//ablodge/amr-utils
pip install ./amr-utils
```

# Data

We release alignment data for AMR Release 3.0 and Little Prince comprising ~60,000 sentences,
as well as 350 sentences with gold alignments in leamr_test.txt and leamr_dev.txt.

We release 4 layers of alignments: **subgraph**, **duplicate subgraph**, **relation**, and **reentrancy** alignments. 

For AMR Release 3.0 and Little Prince, as well as our gold test and dev data we release:

- `<corpus>.subgraph_alignments.json`: Each **subgraph** alignment maps a DAG-shaped subgraph to a single span. We also include **duplicate subgraph** alignments in this layer with the alignment type "dupl-subgraph". Duplicate subgraph alignments are used to represent phenomena such as ellipsis where some semantic content in the AMR is unpronounced but whose semantics is determined by another AMR subgraph.
- `<corpus>.relation_alignments.json`: Each **relation** alignment maps a span to a collection of _external_ edges, where each edge is between two subgraphs aligned in the previous layer. These alignments include argument structures (gave => :ARG0, :ARG1, :ARG2) and single relation alignments (when => :time).
- `<corpus>.reentrancy_alignments.json`: Each **reentrancy** alignment maps a reentrant edge to the span which "triggers" that reentrancy, and is classified with a reentrancy type to account for phenomona like coreference, control, and coordination. 


We also release `<corpus>.spans.json` which species the spans for each sentence, grouping together tokens which are named entities or multiword expressions.


## JSON Format
Alignments are released as JSON files.

To read alignments from a JSON file do:
```
reader = AMR_Reader()
alignments = reader.load_alignments_from_json(alignments_file)
```


## Get Alignments
Anonymized alignments are stored in the folder `data-release/alignments`. To interpret them, you will need the associated AMR data.


## Get AMR Data
You will first need to obtain AMR Release 3.0 from LDC: [https://catalog.ldc.upenn.edu/LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02). Afterwards you can run the following code to unpack the remainder of the data. Make sure to specify `<LDC parent dir>` as the parent directory of your AMR Release 3.0 data.

```
wget https://amr.isi.edu/download/amr-bank-struct-v1.6.txt -O data-release/amrs/little_prince.txt
python build_data.py <LDC parent dir>
python unanonymize_alignments.py
```


# Run Pre-trained Aligner
For a file of unaligned AMRs `<unaligned amr file>`, you can create alignments by running the following code. The script `nlp_data.py` does necessary preprocessing and may take several hours to run on a large dataset.

```
python nlp_data_fast.py data-release/amrs/ldc+little_prince.txt
python nlp_data.py <unaligned amr file>

python align_with_pretrained_model -T data-release/amrs/ldc+little_prince.txt -M data-release/alignments/ldc+little_prince -t <unaligned amr file>
```

# Train Aligner
You can set `<train file>` to 'data-release/amrs/ldc+little_prince' or some other AMR file name. The script `nlp_data.py` does necessary preprocessing and may take several hours to run on a large dataset.

```
python nlp_data.py <train file>.txt

python train_subgraph_aligner.py -T <train file>.txt
mv <train file>.subgraph_alignments.epoch4.json <train file>.subgraph_alignments.json 
python train_relation_aligner.py -T <train file>.txt
mv <train file>.relation_alignments.epoch4.json <train file>.relation_alignments.json 
python train_reentrancy_aligner.py -T <train file>.txt
mv <train file>.reentrancy_alignments.epoch4.json <train file>.reentrancy_alignments.json 
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
