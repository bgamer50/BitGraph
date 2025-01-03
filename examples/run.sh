#!/bin/bash

# tune llm combined/none
python3 construct.py \
    --property_storage pinned \
    --fname /opt/data/para_with_hyperlink.jsonl \
    --truth_fname /opt/data/data/train.json \
    --embeddings_dir /opt/workdir/bitgraph/data/rag/roberta \
    --embedding_type roberta \
    --w2v_path /opt/data/GoogleNews-vectors-negative300.bin.gz \
    --stage finetune_llm \
    --query_file query_v4_roberta.jsonl \
    --query_search_file query_search.json \
    --train_size 16384 \
    --rag_type combined \
    --llm_model openlm-research/open_llama_3b_v2 \
    --llm_mparams 3 \
    --num_epochs 2 > output1.txt

# tune llm direct
 python3 construct.py \
    --property_storage pinned \
    --fname /opt/data/para_with_hyperlink.jsonl \
    --truth_fname /opt/data/data/train.json \
    --embeddings_dir /opt/workdir/bitgraph/data/rag/roberta \
    --embedding_type roberta \
    --w2v_path /opt/data/GoogleNews-vectors-negative300.bin.gz \
    --stage finetune_llm \
    --query_file query_v4_roberta.jsonl \
    --query_search_file query_search.json \
    --train_size 16384 \
    --rag_type direct \
    --llm_model openlm-research/open_llama_3b_v2 \
    --llm_mparams 3 \
    --num_epochs 2 > output2.txt

# train llm combined
python3 construct.py \
    --property_storage pinned \
    --fname /opt/data/para_with_hyperlink.jsonl \
    --truth_fname /opt/data/data/train.json \
    --embeddings_dir /opt/workdir/bitgraph/data/rag/roberta \
    --embedding_type roberta \
    --w2v_path /opt/data/GoogleNews-vectors-negative300.bin.gz \
    --stage train \
    --query_file query_v4_roberta.jsonl \
    --query_search_file query_search.json \
    --train_size 16384 \
    --rag_type combined \
    --llm_model openlm-research/open_llama_3b_v2 \
    --llm_mparams 3 \
    --mlp_out_channels 3200 \
    --llm_params llm_combined_tuned_10_28.pt \
    --num_epochs 2 > output3.txt

# train llm direct
python3 construct.py \
    --property_storage pinned \
    --fname /opt/data/para_with_hyperlink.jsonl \
    --truth_fname /opt/data/data/train.json \
    --embeddings_dir /opt/workdir/bitgraph/data/rag/roberta \
    --embedding_type roberta \
    --w2v_path /opt/data/GoogleNews-vectors-negative300.bin.gz \
    --stage train \
    --query_file query_v4_roberta.jsonl \
    --query_search_file query_search.json \
    --train_size 16384 \
    --rag_type direct \
    --llm_model openlm-research/open_llama_3b_v2 \
    --llm_mparams 3 \
    --mlp_out_channels 3200 \
    --llm_params llm_direct_tuned_10_29.pt \
    --num_epochs 2 > output4.txt

# test llm combined
python3 construct.py \
    --property_storage pinned \
    --fname /opt/data/para_with_hyperlink.jsonl \
    --truth_fname /opt/data/data/train.json \
    --embeddings_dir /opt/workdir/bitgraph/data/rag/roberta \
    --embedding_type roberta \
    --w2v_path /opt/data/GoogleNews-vectors-negative300.bin.gz \
    --stage test \
    --query_file query_v4_roberta.jsonl \
    --query_search_file query_search.json \
    --train_size 16384 \
    --rag_type combined \
    --llm_model openlm-research/open_llama_3b_v2 \
    --llm_mparams 3 \
    --mlp_out_channels 3200 \
    --gnn_llm_params gnn_llm_trained_10_30.pt \
    --num_epochs 2 > output5.txt

# test llm direct
python3 construct.py \
    --property_storage pinned \
    --fname /opt/data/para_with_hyperlink.jsonl \
    --truth_fname /opt/data/data/train.json \
    --embeddings_dir /opt/workdir/bitgraph/data/rag/roberta \
    --embedding_type roberta \
    --w2v_path /opt/data/GoogleNews-vectors-negative300.bin.gz \
    --stage test \
    --query_file query_v4_roberta.jsonl \
    --query_search_file query_search.json \
    --train_size 16384 \
    --rag_type direct \
    --llm_model openlm-research/open_llama_3b_v2 \
   --llm_mparams 3 \
    --mlp_out_channels 3200 \
    --llm_params llm_direct_trained.pt \
    --num_epochs 2 > output6.txt