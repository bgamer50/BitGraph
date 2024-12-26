# QRAG
## Overview
QRAG is a RAG implementation based on G-Retriever designed for handling large knowledge graphs
and complex graph queries.  QRAG bridges the gap between traditional graph analytics and LLMs
by enabling the use of graph queries for information retrieval.  This folder contains
a variety of RAG examples; the main code for QRAG is contained in `construct.py`.

## Reproducing QRAG Results
The results in the paper "QRAG: Using Learnable Graph Queries for Retrieval Augmented Generation" 
were obtained by running the files in this repository.  Users can run the main QRAG benchmarks,
which include comparison to G-Retriever and a non-RAG solution.

### Installation
Running QRAG requires BitGraph and FAISS[https://github.com/facebookresearch/faiss].  To install
FAISS, we recommend building from source as described in their GitHub repository.

To install BitGraph, you can run `build.sh` which will build both the C++ libraries and Python
extensions.  BitGraph requires Maelstrom[https://github.com/bgamer50/maelstrom] and Gremlin++[https://github.com/bgamer50/gremlin-]; instructions for building those libraries
are included in those repositories.  The built libraries should be visible to BitGraph.  The simplest way to do this is by cloning Maelstrom and Gremlin++
into the same directory where BitGraph was cloned (i.e. you may have something like `/opt/code/bitgraph`, `/opt/code/maelstrom`, and `/opt/code/gremlin++`).

### Running
The Python script `construct.py` can run a variety of benchmarks and operations.  The standard benchmarks are shown in `run.sh`.  There is also a visualization
model in `construct.py` which will output the subgraphs extracted by QRAG.  This is very helpful for debugging.  Each stage is described in detail in the QRAG paper.

Prior to running `construct.py`, you will need to generate embeddings, either word2vec (w2v) embeddings, or roberta embeddings.  All benchmarks in the QRAG paper were
run using roberta embeddings, but these are very expensive to generate.  The `preprocess.py` Python script will generate the correct embeddings.  This script must be run
twice, once for sentences, and once for articles.

### Data
There are two key sources of data required to run the benchmarks for QRAG.  The first is the 2WikiMultiHopQA[https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file] dataset.
You will need the latest version of `para_with_hyperlink.zip` which contains `para_with_hyperlink.jsonl`.  This file is provided as the `fname` argument to `construct.py` and as
the `fname_in` argument to `preprocess.py`.  The second is the ground truth data, contained in `data_ids_april7.zip`.  That zip file contains `train.json` and `test.json`, which
are provided to the `truth_fname` argument in `construct.py`.

If you are using w2v embeddings, you will also need to provide the w2v dictionary.  For the QRAG paper, we used the file `GoogleNews-vectors-negative300.bin.gz` which is available
from Kaggle[https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300].

## Questions
For any issues or questions, we recommend making a GitHub issue in the BitGraph repository.