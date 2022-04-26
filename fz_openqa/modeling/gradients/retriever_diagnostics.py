import math
from typing import Dict
from typing import Optional

import torch
from torch import Tensor


@torch.no_grad()
def retriever_diagnostics(
    *,
    retriever_score: Tensor,
    retrieval_score: Optional[Tensor],
    retrieval_rank: Optional[Tensor],
    match_score: Optional[Tensor] = None,
    document_ids: Optional[Tensor] = None,
    reader_score: Optional[Tensor] = None,
    retriever_agg_score: Optional[Tensor] = None,
    retriever_log_p_dloc: Optional[Tensor] = None,
    **kwargs,
) -> Dict:
    """
    Compute diagnostics for the rank of the retrieved documents.

    NB: `retriever_probs` corresponds to the probs of the trained model whereas
    `retrieval_*` correspond to the probs of the model used for indexing.
    """
    output = {}

    retriever_score = retriever_score.clone().detach()
    output["retriever/score-std"] = retriever_score.std()
    output["retriever/score-min-max"] = retriever_score.max() - retriever_score.min()

    retriever_log_probs = retriever_score.log_softmax(dim=-1)
    retriever_probs = retriever_log_probs.exp()

    # arg_i(cdf=p)
    sorted_probs = retriever_probs.sort(dim=-1, descending=True).values
    cdf = sorted_probs.cumsum(dim=-1)
    for p in [0.5, 0.9]:
        arg_cdf_90 = 1 + (cdf - p).abs().argmin(dim=-1)
        output[f"retriever/arg_cdf_{int(100 * p)}"] = arg_cdf_90.float().mean()

    # entropy `H(p(D | q, A))`
    retriever_entropy = -(retriever_probs * retriever_log_probs).sum(dim=(1, 2))
    output["retriever/entropy"] = retriever_entropy.mean()

    # entropy H(p(d))
    if retriever_agg_score is not None:
        log_nq = math.log(retriever_agg_score.size(0))
        log_p_d = (retriever_agg_score.log_softmax(dim=-1) - log_nq).logsumexp(dim=0)
        retriever_entropy_agg = -(log_p_d.exp() * log_p_d).sum(dim=-1)
        output["retriever/entropy_agg"] = retriever_entropy_agg.mean()

    # agg. density of document locations (MaxSim)
    if retriever_log_p_dloc is not None:
        H_retriever_agg = -(retriever_log_p_dloc.exp() * retriever_log_p_dloc).sum(dim=-1)
        output["retriever/maxsim_entropy"] = H_retriever_agg.mean()

    # KL (retriever || U)
    if retrieval_score is not None:
        #  truncate `retrieval_scores` to avoid `NaN` and compute `log r(D | q, A)`
        M = retrieval_score.max(dim=-1, keepdim=True).values
        retrieval_score = retrieval_score - M
        retrieval_score = retrieval_score.clip(min=-1e6)
        retrieval_log_probs = retrieval_score.log_softmax(dim=-1)

        # `KL( p(D|q, A) || r(D|q, A) )`
        kl_div = retriever_probs * (retriever_log_probs - retrieval_log_probs)
        kl_div = kl_div.sum(dim=(1, 2))
        output["retriever/kl_div"] = kl_div.mean()

    # retrieval rank info
    if retrieval_rank is not None:
        # retrieval rank weighted by the probability of the retrieved document
        weighted_rank = retriever_probs * retrieval_rank
        output["retriever/weighted_rank"] = weighted_rank.sum(-1).mean()

        # rank of the most likely document
        top_idx = retriever_probs.argmax(dim=-1).unsqueeze(-1)
        top_rank = retrieval_rank.gather(dim=-1, index=top_idx)
        output["retriever/top_rank"] = top_rank.float().mean()

        # min-max of the retrieval rank
        output["retrieval/n_samples"] = retrieval_rank.size(-1)
        output["retrieval/max_sampled_rank"] = retrieval_rank.max().float()
        output["retrieval/min_sampled_rank"] = retrieval_rank.min().float()

    # match score diagnostics
    if match_score is not None:
        match_logits = (match_score > 0).float().log_softmax(dim=-1)
        kl_relevance = retriever_probs * (retriever_log_probs - match_logits)
        kl_relevance = kl_relevance.sum(dim=(1, 2))
        output["retriever/kl_relevance"] = kl_relevance.mean()

    # diversity of the retrieved documents
    if document_ids is not None:
        unique_ids = document_ids.view(-1).unique()
        output["retrieval/n_unique_docs"] = unique_ids.size(0)
        prop_unique_docs = unique_ids.size(0) / math.prod(document_ids.shape)
        output["retrieval/prop_unique_docs"] = prop_unique_docs

        output["retriever/max-doc-id"] = document_ids.max()
        output["retriever/min-doc-id"] = document_ids.min()

    # reader score diagnostics
    if reader_score is not None:
        reader_score_ = reader_score - reader_score.mean(dim=-1, keepdim=True)
        retriever_score_ = retriever_score - retriever_score.mean(dim=-1, keepdim=True)
        scores_diff = (retriever_score_ - reader_score_).pow(2).mean()
        output["retriever/scores_diff"] = scores_diff

    return output
