# Enhancing Dual-Target Cross-Domain Recommendation via Similar User Bridging

[Overview](#overview) | [Installation](#installation) | [Dataset](#dataset) | [Folder Structure](#folder-structure) | [How to Run Model](#how-to-run-model)

## Overview

Dual-target cross-domain recommendation aims to mitigate data sparsity and achieve mutual enhancement across two domains through bidirectional knowledge transfer. Mainstream approaches in this field primarily leverage overlapping user representations to establish cross-domain connections. However, in many real-world scenarios, overlapping data is extremely limited—or even entirely absent—significantly diminishing the effectiveness of these methods. To address this challenge, we propose \textbf{SUBCDR}, a novel framework that leverages large language models (LLMs) to bridge similar users across domains, thereby enhancing dual-target cross-domain recommendation. Specifically, we introduce a Multi-Interests-Aware Prompt Learning mechanism that enables LLMs to generate comprehensive user profiles, disentangling domain-invariant interest points while capturing fine-grained preferences. Then, we construct intra-domain bipartite graphs from user-item interactions and an inter-domain heterogeneous graph that links similar users across domains. Subsequently, to facilitate effective knowledge transfer, we employ Graph Convolutional Networks (GCNs) for intra-domain relationship modeling and design an Inter-domain Hierarchical Attention Network (InterHAN) to facilitate inter-domain knowledge transfer through similar users, learning both shared and specific user representations. Extensive experiments on seven public datasets demonstrate that SUBCDR outperforms state-of-the-art cross-domain recommendation algorithms and single-domain recommendation methods. The code is anonymously open-sourced at \href{https://anonymous.4open.science/r/SUBCDR-966}{https://anonymous.4open.science/r/SUBCDR-966} for review.

## Installation

Create a python 3.10 environment and install dependencies:

conda create --n SUBCDR python=3.10
source activate SUBCDR

pip install -r requirements.txt

## Dataset
The dataset used in the experiment is Amazon review 2018, which you can download from this link:https://nijianmo.github.io/amazon/index.html. Please download the review and metadata for the corresponding dataset. Then you can execute Then you can execute three files data_raw_before.py, filter.py, process.py and crossaug_test_dataprocess.py in order. You will get train, tast, valid data and user, item index data.

## Folder Structure
<pre> ├─CUT
│  ├─data
│  │      dataloader.py
│  │      dataset.py
│  │      utils.py
│  │      __init__.py
│  │
│  ├─model
│  │  │  crossdomain_recommender.py
│  │  │  __init__.py
│  │  │
│  │  └─cross_domain_recommender   #baseline in recbole and subcdr
│  │          cmf.py
│  │          conet.py
│  │          cut.py
│  │          deepapf.py
│  │          dtcdr.py
│  │          emcdr.py
│  │          han_conv.py
│  │          natr.py
│  │          sscdr.py
│  │          subcdr.py
│  │
│  ├─properties
│  │  │  overall.yaml
│  │  │
│  │  ├─dataset
│  │  │      sample.yaml
│  │  │
│  │  └─model 
│  │          BiTGCF.yaml
│  │          CUT.yaml
│  │          DTCDR.yaml
│  │          SUBCDR.yaml
│  │
│  ├─quick_start
│  │      quick_start.py
│  │      __init__.py
│  │      __init__.pyc
│  │
│  ├─sampler
│  │      crossdomain_sampler.py
│  │      __init__.py
│  │
│  ├─trainer
│  │      trainer.py
│  │      __init__.py
│  │
│  └─utils
│          enum_type.py
│          utils.py
│          __init__.py
│
├─dataset
│  └─Amazon
│      │  data_raw_before.py  #data process
│      │  filter.py
│      │  process.py
│      │
│      ├─Amazon_toy  #subcdr processed data
│      │  │  Amazon_toy.test.inter
│      │  │  Amazon_toy.train.inter
│      │  │  Amazon_toy.valid.inter
│      │  │  similarity_tags.txt
│      │  │  test.txt
│      │  │  valid.txt 
│      │
│      └─Amazon_video
│          │  Amazon_video.test.inter
│          │  Amazon_video.train.inter
│          │  Amazon_video.valid.inter
│          │  similarity_aspect_tags.txt
│          │  similarity_tags.txt
│          │  similar_user_10.txt
│          │  summary_onlytop10_similarity_tags.txt
│          │  summary_only_similarity_tags.txt
│          │  tags_edge3.txt
│          │  tags_edge4.txt
│          │  tags_random10_120_similarity_tags.txt
│          │  test.txt
│          │  valid.txt
│          │
│
├─LLM #LLM parts
│      0.amazon_prompt.ipynb
│      0.movielens_prompt.ipynb
│      1.vllm_generate.py
│      2.tag_error.ipynb
│      3.tag_mapped_v11.ipynb
│      4.edge_generate.ipynb
│      5.edge_process.ipynb
│      video_prompt_v11.txt
│      vllm_infer.py
│
└─scripts
        run-arts.sh
        run-cell-1.sh
        run-cell.sh
        run-gate.sh
        run-Music.sh
        run-reg.sh
        run-sport.sh
        run-video.sh
        run_gcn.sh </pre>

## How to run SUBCDR
1. download Llam3-8b model
2. run jupiter files in LLM folder in order
3. if you just want to test subcdr, we provide Toy-Video datasets and its similar user's file. You can execute "./run-video.sh" to reproduce our results. We used one A100. If you reduce the data of the similar user, you can try our code with 3090. 
