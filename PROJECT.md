# Medical Open Domain Question Answering

Open Domain Question Answering (OpenQA) has improved significantly in the last few years thanks to the wide availability of powerful deep language models such as BERT.

FindZebra aims at improving the diagnosis of rare disease using information retrieval. Whereas the current solution relies on classic information retrieval tools (non-ML), there is an opportunity to improve FindZebra latest deep learning techniques.

The FindZebra corpus contains 30,658 individual disease descriptions. FindZebra's diagnosis tool consists of retrieving the right disease description (evidence) based on a patient description (query). Therefore FindZebra essentially aims at solving a retrieval problem, which can be framed as Open Domain QA (OpenQA).

The [MedQA](https://arxiv.org/pdf/2009.13081.pdf) dataset, which is now mapped to the FZ corpus will be exploited in this project. The MedQA authors introduced a simple basline Module using a BM25 retriever and a bidirectional GRU reader Module.

In this project, we will 1. attempt to improve upon the baseline on MedQA and 2. improve FindZebra retrieval using OpenQA.

## Notation
* $\mathbf{q}$: query
* $\mathbf{d}$: an evidence document from the corpus
* $\mathbf{a}$: an answer
* $\mathbf{x} = (\mathbf{q}, \mathbf{d}, \mathbf{a})$: a triplet of query, evidence and answer
* $\mathbf{y}_{1:N}$: batch of N vectors $\mathbf{y}_i$
* $\mathcal{C} := \{\mathbf{d}_1, \dots, \mathbf{d}_N\}$: corpus of documents
* $\mathcal{X}$: the embedded text domain: $\mathcal{R}^{T \times h}$
* $[\cdot\ ;\ \cdot]$ concatenation operator
* $\mathbf{h^y}:=\rm{BERT}(\mathbf{y})$: encoding of the variable $\mathbf{y}$
* $\mathbf{h}_i$: the $i$-th component of the vector $\mathbf{h}$.
* $p_\theta(\mathbf{a} | \mathbf{q}, \mathbf{d})$: *reader*
* $p_\phi(\mathbf{d} | \mathbf{q})$: *retriever*



---


## 1. Primer on OpenQA

![](https://i.imgur.com/XD0UY0H.png)

### Reading Comprehension

Reading comprehension (i.e. SQUAD dataset) aims at answering a question $\mathbf{q}$ based on a *known* evidence document $\mathbf{d}$.

BERT-based models [`BERT`, `RoBERTa`, `ALBERT`] hold the [top of the leaderboard](https://paperswithcode.com/sota/question-answering-on-squad20). A linear layer (*head*) is added on top of the BERT Module to extract features that will used to build the *reader* $p_\theta(a | q, d)$.

#### Implementing the *reader* $p_\theta(\mathbf{a} | \mathbf{q}, \mathbf{d})$

The implementation of the reader depends on the formulation of the answering problem.

##### a. Span-based QA

In Span-based QA (e.g. SQUAD), the answer is a span of token within the *evidence* document. Given an evidence $\mathbf{d}$ of length $T$ tokens, the answer $\mathbf{a}$ is defined as the tuple of indexes $(\rm{start}, \rm{stop})$ where $\rm{start} \in [0, T-1]$ and $\rm{stop} \in [\rm{start}, T-1]$. Given the emebddings $S \in \mathcal{R}^d$ and $E \in \mathcal{R}^d$, the *reader* can be defined as:

$$p_\theta(\mathbf{a} | \mathbf{q}, \mathbf{d} ) := p_\theta(\rm{start} | \mathbf{q}, \mathbf{d})\ p_\theta(\rm{stop} | \rm{start}, \mathbf{q}, \mathbf{d}) $$

with

$$p_\theta(\rm{start} | \mathbf{q}, \mathbf{d}) = \frac{e^{S \cdot \mathbf{h_{\rm{start}}}}}{\sum_{j} e^{S \cdot \mathbf{h_{j}}}}, \qquad
p_\theta(\rm{end} | \rm{start}, \mathbf{q}, \mathbf{d}) = \frac{e^{S \cdot \mathbf{h_{\rm{start}}} + E \cdot \mathbf{h_\rm{end}}}}{\sum_{j \geq \rm{start}} e^{S \cdot \mathbf{h _{\rm{start}}} + E \cdot \mathbf{h _j}}}$$

where, for instance, $\mathbf{h} := \rm{BERT}([\mathbf{q}; \mathbf{d}]) \in \mathcal{X}$.

##### b. Multiple-choice QA

In this setting, each question-evidence pair comes with a set of $P$ answer candidates $\mathbf{A} = \{\mathbf{a}_1,\dots,\mathbf{a}_P \}$. The answer $a$ is the index of the right answer within $\mathbf{A}$. Given a $f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} ): \mathbf{q}, \mathbf{d}, \mathbf{a} \in \mathcal{X}^3 \rightarrow \mathcal{R}$, the *reader* is defined as:

$$p_\theta(a | \mathbf{q}, \mathbf{d}, \mathbf{A} ) := \frac{e^{f_\theta(\mathbf{a_a}, \mathbf{q}, \mathbf{d} )}}{\sum_p e^{f_\theta(\mathbf{a_p}, \mathbf{q}, \mathbf{d} )}}$$

---

### Information Retrieval

In Open Domain Question Answering, a large corpus of documents is available, and an information retrieval (IR) system is used to noisily retrieve a set of evidence given the question. The reading comprehension Module thereafter answers the question based on the noisy evidence documents. All IR models rely on a function $\mathrm{Sim}: \mathcal{X}, \mathcal{X} \rightarrow \mathcal{R}$ that to score the documents $\mathbf{d}$ given the query $\mathbf{q}$. Using the $S$, we can formally define the *retriever* as

$$p_\phi(\mathbf{d} | \mathbf{q}) :=  \frac{e^{\mathrm{Sim}(\mathbf{q}, \mathbf{d})}}{\sum_{\mathbf{d'} \in \mathcal{C}} e^{\mathrm{Sim}(\mathbf{q}, \mathbf{d'})}}$$


#### Scoring functions and similarity metrics

The score function is often choosen to be a similarity metric. The simplest ones require comparing document features $y \in \mathcal{R}^d$, for which we define a similarity metric $\mathrm{Sim} : \mathcal{R}^d \times \mathcal{R}^d \rightarrow \mathcal{R}$. Common choices are the dot-product, L2 distance or cosine similarity.


##### Granularity

We can also define more complex similarity metrics, including metrics operating on the token-level features $y \in \mathcal{X}$  (as used in ColBERT). We distinguish:

\begin{align}
\text{document-level metrics:} \qquad& \mathrm{Sim} : \mathcal{R} \times \mathcal{R} \rightarrow \mathcal{R} \\
\text{token-level metrics:} \qquad& \mathrm{Sim} : \mathcal{X} \times \mathcal{X} \rightarrow \mathcal{R}
\end{align}

##### Retrieval

Retrival is performed using maximum similarity / nearest neighbour search:

$$\rm{mss}(q, \mathcal{C}) = \mathop{\mathrm{arg\,min}}_{d \in \mathcal{C} }\ \mathrm{Sim}(\mathbf{h_\rm{CLS}^q}, \mathbf{h_\rm{CLS}^d})$$

#### Related Work

##### [DrQA: Reading Wikipedia to answer Open-domain Questions](https://arxiv.org/pdf/1704.00051.pdf)

*Machine reading at scale* consists of using a non-ML IR system (TF-IDF + bigrams) to retrieve 5 potential evidence documents that are thereafter processed by a BERT-based *reader*. Whereas this paper suggests leveraging BERTs for reading comprehension, the retrieval step remains based on non-ML methods.


##### [DPR: Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)

DPR proposes learning vectors representations of the queries and the documents using BERT (representation at the `<CLS>` token). Each document is represented by a single vector $\mathbf{h_\rm{CLS}^{.}} \in \mathcal{R}^d$ (BERT output at the `<CLS>` token).

By contrast with ORQA that models evidences as a latent variable, DPR exploits supervised learning (using triplets `(query, gold document,  answer)`) and in-batch contrastive learning (using other examples from the batch as negative examples). Given a batch of triplets $x_{1:N}$, the objective is:

$$L(x_{1:N}) := - \frac{1}{N} \sum_{i=1}^N \log \frac{e^{d(\mathbf{h_\rm{CLS}^{q_i}}, \mathbf{h_\rm{CLS}^{d_i}})}}{\sum_{j=1}^N e^{d(\mathbf{h_\rm{CLS}^{q_i}}, \mathbf{h_\rm{CLS}^{d_j}}) }}$$

where the similarity metric $d$ can be choosen as the dot-product, the cosine similairy or the L2 distance. However, DPR assumes the evidence documents are known. When the evidence is unknown, they use BM25 to rank evidences and choose the best match which includes the answer span.

##### [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832.pdf)

Whereas DPR relies on document-level representations ($\mathbf{h_\rm{CLS}} \in \mathcal{R}^d$), current research shows that exploiting *local interactions* is more effective. Local interactions involve comparing the query and the documents using token-level representations ($\mathbf{h} \in \mathcal{R}^{T \times m}, m \leq d$). However, token-level metrics are often computationally prohibitive.

ColBERT introduces a *late interaction* Module based on the `MaxSim` operator that allows to scalably compare documents and query based on local document representations. The `MaxSim` operator is defined as

$$ \rm{MaxSim}(q, p):=\sum_{i \in\left[\left|\mathbf{h^q}\right|\right]} \max _{j \in\left[\left|\mathbf{h^{d}}\right|\right]} d(\mathbf{h^q}_{i}, \mathbf{h^d}_{j})$$

where the similarity metric $d$ can be choosen as the dot-product, the cosine similairy or the L2 distance. The representations $E_.$ are obtained using:

\begin{aligned}
\mathbf{h^q} &:=\text { Normalize }\left(\operatorname{CNN}\left(\operatorname{BERT}\left([Q] q_{0} q_{1} \ldots q_{l} \# \# \ldots \# \right)\right)\right) \\
\mathbf{h^{d}} &:=\text { Filter }\left(\text { Normalize }\left(\operatorname{CNN}\left(\operatorname{BERT}\left([D] d_{0} d_{1} \ldots d_{n} \right)\right)\right)\right)
\end{aligned}

##### [ORQA: Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/pdf/1906.00300.pdf)

> todo

##### [REALM: Retrieval-Augmented Language Module Pre-Training](https://arxiv.org/abs/2002.08909)

> todo

## 2. Datasets

| Dataset                      | FZ Corpus |    MedQA     |   FZxMedQA   |
|:---------------------------- |:---------:|:------------:|:------------:|
| Corpus                       | 30k docs  | 12.7M tokens | 12.7M tokens |
| Evidences                    |     -     |      -       |   4k docs    |
| Questions *without* evidence |     -     |     12k      |     12k      |
| Questions *with* evidence    |     -     |      -       |      2k      |


### [MedQA](https://arxiv.org/pdf/2009.13081.pdf)

The MedQA dataset features 61k pairs (query, answer candidates) collected from the National Medical Board Examination (USA, Mainland China and Taiwan) as well as a large corpus (12.7M tokens (en), 14.7M tokens (Zh)) based in medical textbooks.

Each question features 4 answer candidates, the aim is to select the right answer based on the evidence corpus, although no gold evidence is provided.


### FindZebra Corpus

The FZ corpus contains 30.658 disease descriptions (source document). Each disease is identified by a CUI (Concept Unique Identifier, UMLS Thesaurus).

### Mapping [FZxMedQA](https://docs.google.com/document/d/1e5_9C5sfjRSZEYZ8917o2Z9iBdakYXOCUb5vljj6cBs/edit)

The mapping relies on using MetaMap to map the MedQA answers to a CUI, which can be subsequently joined to the FZ corpus based on the CUI.

#### Dataset Formatting

At first, we focus on the FZxMedQA dataset where the evidences are known (2k). The dataset will be of the following format:

|      query       |      document      |   answer_choices    |     answer_idx     |       rank       | is_negative                            |
|:----------------:|:------------------:|:-------------------:|:------------------:|:----------------:| -------------------------------------- |
| <query text>:str | <passage text>:str | <choices>:List[str] | <answer index>:int | <BM25 rank>:int | <is in gold document>:bool |


#### Dataset overview

| dataset | #samples | #questions | Document length | Stride (Overlap) | Top n retrieval |
|:-------:|:--------:|:----------:|:---------------:|:----------------:|:---------------:|
|  Train  |  52.300  |   2.092    |       100       |        80        |       25        |
|   Val   |  6.200   |    248     |       100       |        80        |       25        |
|  Test   |  7.025   |    281     |       100       |        80        |       25        |
|  -----  |  -----   |   -----    |      -----      |      -----       |      -----      |
|  Train  |  52.300  |   2.092    |       100       |        50        |       25        |
|   Val   |  6.200   |    248     |       100       |        50        |       25        |
|  Test   |  7.025   |    281     |       100       |        50        |       25        |
|  -----  |  -----   |   -----    |      -----      |      -----       |      -----      |
|  Train  |          |   2.092    |       200       |       100        |       25        |
|   Val   |          |    248     |       200       |       100        |       25        |
|  Test   |          |    281     |       200       |       100        |       25        |

Download data [here](https://drive.google.com/drive/folders/1tS-O1Q7mkGHahg675HihogNMCGyrL_gY?usp=sharing).


## 3. Proposal for MedQA

In the MedQA paper, the authors introduced a baseline based on BM25 for the IR step and RoBERTa for the reading comprehension step.

We aim at improving the existing along three axes: 1. using ColBERT for the IR step, 2. using the FZxMedQA (triplets) to aid the optimization of the IR Module, 3. using BERT/ColBERT for the *reding* step

### A. Reader $p_\theta(\mathbf{a} | \mathbf{q}, \mathbf{d})$

By contrast with SQUAD, MedQA involves a multiple-choice question with P=4 candidate answers $\mathbf{A} = \{\mathbf{a}_1,\dots,\mathbf{a}_P \}$. As detailed in the previous section, the *reader* is defined as

$$p_\theta(\mathbf{a_i} | \mathbf{q}, \mathbf{d}, \mathbf{A} ) := \frac{\exp(f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} ))}{\sum_p \exp(f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} ))}$$

However, we remain free to choose our own implementation for $f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} )$. BERT (a simpler BiGRU Module can be substitued for development)  can be used to compute the contextualized reresentations $\mathbf{h^d} := \rm{BERT}(\mathbf{d})$ and $\mathbf{h^{qa_i}} := \rm{BERT}(\mathbf{qa_i})$. We are free to choose an interaction Module as:

\begin{align}
\text{a. document level Module:}\qquad f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} ) & := d(\mathbf{h^{qa_i}_\rm{CLS}}, \mathbf{h^d_\rm{CLS}}) \\
\text{b. token level Module       :}\qquad f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} ) & := \rm{MaxSim}(\mathbf{h_{qa_i}}, \mathbf{h_d})
\end{align}

At the token-level, the `MaxSim` can alternatively be replaced by a dense version (attention) since the *reader* is not constrained by scalability.

### B. Retriever  $p_\phi(\mathbf{d} | \mathbf{q})$

The Information Retrieval Module consists of a language representation Module (BERT backbone) and an interaction Module (e.g. `MIPS`, `MaxSim`).

#### Backbone

 Following the current literature, we implement the IR Module using a masked language Module (BERT) as a backbone.
Ideally, we would prefer to use a lighter Module such as a medical [distilBERT](https://arxiv.org/abs/1909.10351).

#### Interaction Module
To guarantee computation tractability, we will choose to implement the ColBERT retriever, for which  the `MaxSim` operator wil be implemented using [Faiss](https://github.com/facebookresearch/faiss).

#### Optimization

Optimizing the IR Module is challenging due to the size of the corpus and the backbone Module.

**a. Supervised using FZxMedQA** When the evidence document is known, supervised learning can be leveraged for optimizing the retriever. In the case of a <u>single evidence document</u>, we use the same approach as described in DPR, so the evidence likelihood can be expressed using:

$$p_\phi(d | q) = \frac{\rm{Sim}(d, q)}{\sum_{d' \in \mathcal{C}} \rm{MaxSim}(d', q) } \approx \frac{\rm{Sim}(d, q)}{\sum_{d' \in \rm{batch}} \rm{Sim}(d', q) }$$

**b. Unsupervised** In the original MedQA dataset, the gold evidence document is unknown. Therefore, we need to rely on unsupervised methods, and handle documents as a latent variable (ORQA, REALM). In that scenario, the answering Module is:

$$p(a|q,\mathcal{C}) = \sum_{d\in \mathcal{C}}\ p_\theta(a|q, d)\ \  p_\phi(d|q)$$

where $p_\theta(a|q, d)$ is the document reader and $p_\phi(d|q)$ is the retriever. Marginalizing over $\mathcal{C}$ is intractable, hence the challenge is about finding an efficient way to approximate $\mathbb{E}_{p(d)}\left[p_\theta(a|q, d)\ p_\phi(d|q) \right]$ or finding an effective way to approximate the gradient for the parameter $\phi$. ORQA suggests first bootstrapping the learning of $\phi$ using the *Inverse Cloze Task* (ICT), second sampling top-k samples from retriever to approximate (I ommit the details about predicting the answer tokens vs. predicting the spans):

$$p(a|q,\mathcal{C}) \approx \sum_{d \in \rm{topk}\left(p_\phi(d | q)\right)}\ p_\theta(a|q, d)\ \  p_\phi(d|q)$$

In any case, this most likely requires to sample multiple evidence documents for each query, which can't be done without massive compute.

**c. Semi-supervised Learning** The unsupervised approach can also be coupled with the supervised objective, jointly or sequentially.

## 4. [Development Plan](https://github.com/vlievin/fz-openqa/projects/1)

### [Milestone #1 - project and dataset setup](https://github.com/vlievin/fz-openqa/milestone/1)

- [x] Set up a project with `Lightning`, `Hydra`.
- [x] Implement the FZxMedQA using HuggingFace's `datasets` library

### [Milestone #2 - basic retriever/reader implementation](https://github.com/vlievin/fz-openqa/milestone/2)

- [x] Implement the basic `reader` Module and train using the gold document
- [x] Implement the basic `retriever` Module and train using the supervised objective
- [ ] Implement the full datset ranking using the trained retriever and `faiss`

### [Milestone #3 - supervised OpenQA](https://github.com/vlievin/fz-openqa/milestone/3)

- [x] Implement the OpenQA Module (`reader+retriever`) and train using the gold document (`reader`) and supervised objective (`retriever`)
- [x] Implement the Corpus dataset using `datasets`
- [ ] Perform end-to-end evaluation (top-k accuracy)
- [ ] Analyze and report results on the FZxMedQA dataset
- [ ] Compare with the `BM25 + reader` baseline

### [Milestone #4 - Colbert Implementation](https://github.com/vlievin/fz-openqa/milestone/4)

- [ ] Implement the OpenQA `retriever` using Colbert (training)
- [ ] Implement the OpenQA `retriever` using Colbert (ranking)
- [ ] Compare with the non-Colbert OpenQA version

### [Milestone #5 - Semi-Supervised OpenQA](https://github.com/vlievin/fz-openqa/milestone/5)

- [ ] Train the Module without FZxMedQA supervision


## 5. Ideas / Future Work


### 1. Distilled BioBERT

> todo

### 2. Variational OpenQA

$$\log p(a | q) \geq \mathbb{E}_{q_\phi(d | q, a)} \left[ \log \frac{p_\theta(a, d | q)}{q_\phi(d | a, q)} \right]$$

where $p_\theta(d | q)$ shares most parameters with $q_\phi(d | a, q)$. Importance-Weighted Sampling can be used on top of that for variance reduction of the gradient using OVIS.

### 3. use [SciSpacy](https://allenai.github.io/scispacy/) to label the MedQA corpus

SciSpacy provides a NER Module for TUIs.

### 4. Score full documents

* [Birch](https://arxiv.org/pdf/1903.10972.pdf)

### 5. Improve negative exmaples:
* [RocketQA](https://arxiv.org/pdf/2010.08191.pdf)

### 6. [Relevance-guided supervision](https://arxiv.org/pdf/2007.00814.pdf)

> todo: read it

### 7. Explicit the Module for scoring all passages within a document

> todo: case where there are multiple evidence passages $f(\rm{doc}) = \rm{max}_{p \in \rm{doc}} f(p)$


# Appendix

## Definitions

### Scoring functions and similarity metrics

The score function is often choosen to be a similarity metric. The simplest ones require comparing document features $y \in \mathcal{R}^d$, for which we define a similarity metric $d : \mathcal{R}^d \times \mathcal{R}^d \rightarrow \mathcal{R}$. Common choices are:

* dot-product:
    $$d^{\rm{dot}}(x,y) := <x,y> := \sum\limits_{|y|} x_i \cdot y_i = x^Ty$$
* cosine similarity:
    $$d^{\rm{Cosine}}(x,y) := \frac{<x,y>}{\|x\|_2\|y\|_2}$$
* L2 distance:
$$d^{L2}(x,y) := \|y - x\|_{2} := \sqrt{ \sum\limits_{|y|} y_i^2 - x_i^2 }$$


## Related Work

### MedQA

The reader Module $f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} )$ is implemented as:

\begin{align}
f_\theta(\mathbf{a_i}, \mathbf{q}, \mathbf{d} )  := W_{1}\left(\tanh \left(W_{2} \mathbf{h}\right)\right) \in \mathbb{R}^{1} \\
\mathbf{h}:=\left[\mathbf{h}^{\mathbf{d}} ; \mathbf{h}^{\mathbf{q} \mathbf{a}_{\mathbf{i}}} ; \mathbf{h}^{\mathbf{d}} \cdot \mathbf{h}^{\mathbf{q a_i}} ;\left|\mathbf{h}^{\mathbf{d}}-\mathbf{h}^{\mathbf{q} \mathbf{a}_{\mathbf{i}}}\right|\right]\\
\mathbf{h^e} := \rm{maxpool}(h_\theta(\mathbf{d})), \quad \mathbf{h^{qa_i}} := \rm{maxpool}(h_\theta([\mathbf{q} ; \mathbf{a_i}]))
\end{align}

where $h_\theta(\cdot) \in \mathcal{X}$ is output of a bidirectional GRU.

### [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832.pdf)

Whereas DPR relies on document-level representations ($\mathbf{h_\rm{CLS}} \in \mathcal{R}^d$), current research shows that exploiting *local interactions* is more effective. Local interactions involve comparing the query and the documents using token-level representations ($\mathbf{h} \in \mathcal{R}^{T \times m}, m \leq d$). However, token-level metrics are often computationally prohibitive.

ColBERT introduces a *late interaction* Module based on the `MaxSim` operator that allows to scalably compare documents and query based on local document representations. The `MaxSim` operator is defined as

$$ \rm{MaxSim}(q, p):=\sum_{i \in\left[\left|\mathbf{h^q}\right|\right]} \max _{j \in\left[\left|\mathbf{h^{e}}\right|\right]} d(\mathbf{h^q}_{i}, \mathbf{h^e}_{j})$$

where the similarity metric $d$ can be choosen as the dot-product, the cosine similairy or the L2 distance. The representations $E_.$ are obtained using:

\begin{aligned}
\mathbf{h^q} &:=\text { Normalize }\left(\operatorname{CNN}\left(\operatorname{BERT}\left([Q] q_{0} q_{1} \ldots q_{l} \# \# \ldots \# \right)\right)\right) \\
\mathbf{h^{e}} &:=\text { Filter }\left(\text { Normalize }\left(\operatorname{CNN}\left(\operatorname{BERT}\left([D] e_{0} e_{1} \ldots e_{n} \right)\right)\right)\right)
\end{aligned}

For short queries, the query is expanded by appending the query tokens with the special token "#".

###### Optimization

The `MaxSim` operator and can be used to compute the $L_{\mathbf{x}_{1:N}}$ and the resulting loss function is differentiable.

###### Retrieval using `Faiss`

Computing the `Maxsim` operator over millions of documents is prohibitevly expensive, and the `MaxSim` is not implemented in similarity search libraries like `Faiss`, only vector-level similarities ($d:\mathcal{R}^{d} \times \mathcal{R}^d \rightarrow \mathcal{R}$ are available). Therefore, ColBERT introduce a two-steps retrieval that relies on indexing the token contextualized representations $\mathbf{h}_i$ instead of the document representations $\mathbf{h}$. An index between token (indexed in `Faiss`) and document is kept.

1. *Filtering*: retrieve the top-$k'$ ($k' = k/2$) closest tokens for the $N_q$ query tokens. The document-token index allows to return $K \leq N_q \times k'$ unique documents.

2. *Refining*: Apply the `MaxSim` operator to the $K$ documents.

> todo: detail IVFPQ indexing

###### Prototyping considerations

The FZxMedQA datasets features only 2k lines. We do not need the efficient `faiss` retriever in that case.

###### Implementation

ColBERT relies on triplets $\mathbf{x} = (\mathbf{q}, \mathbf{d}, \mathbf{a})$ (supervised setup). Pseudo-code for trainig and retrieval:

1. Training:
```python=
hq = ColBERT(queries)
hd = ColBERT(docs)
score = MaxSim(hq, hd)
targets = range(len(score))
loss = cross_entropy(score, targets)
loss.backward()
```


2. Retrieval

```python=
# step 0: index
doc_tokens: Iterable[LongTensor] = tokenize(corpus) #  each Tensor of shape [l, d]
tok2doc = {tok_idx: doc_idx for doc_idx, tokens in enumerate(doc_tokens) \
                            for tok_idx, tok in enumerate(tokens)}
faiss_index = faiss.index((tok for tokens in doc_tokens for tok in tokens))

# step 1: [filtering] get K closest documents
k' = k/2
K_docs = set()  # use a set to remove duplicates
for hq in query_tokens:
    tok_idxs = faiss_index.search(hq, k=k')
    K_doc_idxs |= {tok2doc[t] for t in tok_idxs}
# retrive the K documents
K_docs = [corpus[i] for i in K_doc_idxs]

# step 2: [refining] apply MaxSim
top_k_docs = arg_topk(MaxSim(query_tokens, K_docs))
```


---
---
---
---

# Clipboard

#### a. Implement datasets

* MedQA
* FZxMedQA

#### b. experimental framework

* config management using [Hydra](https://hydra.cc/docs/intro/)
* training management using [PyTorch Lightning](https://www.pytorchlightning.ai)
* logging using [Weights & Biases](https://wandb.ai)

#### b. Baseline Implementation

* TF-IDF retriever
* BM25 retriever
* Reader Module (as implemented in MedQA)

#### c. Implementation

* `MIPS` using `faiss`
* `MaxSim` using `faiss`
* IR opt: Dense Passage Retrieval applied to FZxMedQA (where triplets are available)
* IR opt: unsupervised

#### d. potential steps

* BioBERT distillation
