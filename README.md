<div align="center">

> **Warning** We are currently working on extending this code to other NLP tasks at [VodLM](https://github.com/VodLM).

 > **Note** Find an implementation of the sampling methods and the VOD objective [VodLM/vod](https://github.com/VodLM/vod).
 
# Variational Open-Domain Question Answering

<p align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/variational-open-domain-question-answering/multiple-choice-question-answering-mcqa-on-21)](https://paperswithcode.com/sota/multiple-choice-question-answering-mcqa-on-21?p=variational-open-domain-question-answering)
 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/variational-open-domain-question-answering/question-answering-on-medqa-usmle)](https://paperswithcode.com/sota/question-answering-on-medqa-usmle?p=variational-open-domain-question-answering)

</div>

## Abstract

> Retrieval-augmented models have proven to be effective in natural language processing tasks, yet there remains a lack of research on their optimization using variational inference. We introduce the Variational Open-Domain (VOD) framework for end-to-end training and evaluation of retrieval-augmented models, focusing on open-domain question answering and language modelling. The VOD objective, a self-normalized estimate of the Rényi variational bound, is a lower bound to the task marginal likelihood and evaluated under samples drawn from an auxiliary sampling distribution (cached retriever and/or approximate posterior). It remains tractable, even for retriever distributions defined on large corpora. We demonstrate VOD's versatility by training reader-retriever BERT-sized models on multiple-choice medical exam questions. On the MedMCQA dataset, we outperform the domain-tuned Med-PaLM by +5.3% despite using 2.500x fewer parameters. Our retrieval-augmented BioLinkBERT model scored 62.9% on the MedMCQA and 55.0% on the MedQA-USMLE. Last, we show the effectiveness of our learned retriever component in the context of medical semantic search.

## Setup


1. Setting up the Python environment

```shell
curl -sSL https://install.python-poetry.org | python -
poetry install
```

2. Setting up ElasticSearch

```shell
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.14.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.14.1-linux-x86_64.tar.gz
```
To run ElasticSearch navigate to the `elasticsearch-7.14.1` folder in the terminal and run `./bin/elasticsearch`.

3. Run the main script

```shell
poetry run python run.py
```

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2210.06345,
  doi = {10.48550/ARXIV.2210.06345},
  url = {https://arxiv.org/abs/2210.06345},
  author = {Liévin, Valentin and Motzfeldt, Andreas Geert and Jensen, Ida Riis and Winther, Ole},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.7; H.3.3; I.2.1},
  title = {Variational Open-Domain Question Answering},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Credits

The package relies on:

* [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to simplify training management including
  distributed computing, logging, checkpointing, early stopping, half precision training, ...
* [Hydra](https://hydra.cc/docs/intro/) for a clean management of experiments (setting hyper-parameters, ...)
* [Weights and Biases](https://wandb.ai) for clean logging and experiment tracking
* [Poetry](https://python-poetry.org/) for stricter dependency management and easy packaging
* The original template was copied form [ashleve](https://github.com/ashleve/lightning-hydra-template)

