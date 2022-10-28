import math
from typing import Dict
from typing import Optional

import torch
from torch import Tensor


def flat_scores_no_nans(scores):
    scores_ = scores.view(-1)
    scores_ = scores_[(~scores_.isnan()) & (~scores_.isinf())]
    return scores_


@torch.no_grad()
def retriever_diagnostics(
    *,
    retriever_score: Tensor,
    proposal_score: Optional[Tensor],
    proposal_rank: Optional[Tensor] = None,
    doc_ids: Optional[Tensor] = None,
    raw_doc_ids: Optional[Tensor] = None,
    reader_score: Optional[Tensor] = None,
    retriever_agg_score: Optional[Tensor] = None,
    retriever_log_p_dloc: Optional[Tensor] = None,
    share_documents_across_batch: bool = False,
    **kwargs,
) -> Dict:
    """
    Compute diagnostics for the rank of the retrieved documents.

    NB: `retriever_probs` corresponds to the probs of the trained model whereas
    `proposal_*` correspond to the probs of the model used for indexing.
    """
    output = {}

    retriever_score = retriever_score.clone().detach()
    flat_scores = flat_scores_no_nans(retriever_score)
    if len(flat_scores) > 0:
        output["retriever/score-std"] = flat_scores.std()
        output["retriever/score-min-max"] = flat_scores.max() - flat_scores.min()

    # compute p(d | q)
    retriever_log_probs = retriever_score.log_softmax(dim=-1)
    retriever_probs = retriever_log_probs.exp()

    # arg_i(cdf=p)
    if retriever_probs.device == torch.device("cpu"):
        retriever_probs = retriever_probs.to(torch.float32)
    sorted_probs = retriever_probs.sort(dim=-1, descending=True).values
    cdf = sorted_probs.cumsum(dim=-1)
    for p in [0.5, 0.9]:
        arg_cdf_90 = 1 + (cdf - p).abs().argmin(dim=-1)
        output[f"retriever/arg_cdf_{int(100 * p)}"] = arg_cdf_90.float().mean()

    # entropy `H(p(D | q, A))`
    retriever_entropy = -(retriever_probs * retriever_log_probs).sum(dim=-1)
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
    if proposal_score is not None:
        #  truncate `proposal_scores` to avoid `NaN` and compute `log r(D | q, A)`
        proposal_log_probs = proposal_score.log_softmax(dim=-1)

        # entropy `H(p(D | q, A))`
        proposal_entropy = -(proposal_log_probs.exp() * proposal_log_probs).sum(dim=-1)
        output["proposal/entropy"] = proposal_entropy.mean()

        if not share_documents_across_batch:
            # `KL( p(D|q, A) || r(D|q, A) )`
            kl_div = retriever_probs * (retriever_log_probs - proposal_log_probs)
            kl_div = kl_div.sum(dim=-1)
            output["retriever/kl_div"] = kl_div.mean()

    # retrieval rank info
    if proposal_rank is not None:
        if not share_documents_across_batch:
            # retrieval rank weighted by the probability of the retrieved document
            weighted_rank = retriever_probs * proposal_rank
            output["retriever/weighted_proposal_rank"] = weighted_rank.sum(-1).mean()

            # rank of the most likely document
            top_idx = retriever_probs.argmax(dim=-1).unsqueeze(-1)
            top_rank = proposal_rank.gather(dim=-1, index=top_idx)
            output["retriever/top_proposal_rank"] = top_rank.float().mean()

        # min-max of the retrieval rank
        output["proposal/n_samples"] = proposal_rank.size(-1)
        output["proposal/max_sampled_rank"] = proposal_rank.max().float()
        output["proposal/min_sampled_rank"] = proposal_rank.min().float()
        output["proposal/mean_sampled_rank"] = proposal_rank.float().mean()

    # diversity of the retrieved documents
    if doc_ids is not None:
        if raw_doc_ids is None:
            total_ids = math.prod(doc_ids.shape)
        else:
            total_ids = math.prod(raw_doc_ids.shape)
        unique_ids = doc_ids.view(-1).unique()
        output["proposal/n_unique_docs"] = unique_ids.size(0)
        prop_unique_docs = unique_ids.size(0) / total_ids
        output["proposal/prop_unique_docs"] = prop_unique_docs

        # ddoc ids
        output["proposal/max-doc-id"] = doc_ids.max()
        output["proposal/min-doc-id"] = doc_ids.min()

    # reader score diagnostics
    if reader_score is not None:
        reader_score_ = reader_score - reader_score.mean(dim=-1, keepdim=True)
        retriever_score_ = retriever_score - retriever_score.mean(dim=-1, keepdim=True)
        scores_diff = (retriever_score_ - reader_score_).pow(2).mean()
        output["retriever/scores_diff"] = scores_diff

    return output
