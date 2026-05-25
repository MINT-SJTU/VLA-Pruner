"""Utilities for VLA-Pruner style token selection on OpenVLA-OFT.

The OpenVLA-OFT path predicts a full action chunk in one forward pass, so the
useful attention signals are prompt/semantic tokens and action placeholder
tokens, not autoregressive action-generation probabilities.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch

from experiments.robot.vla_cache_utils import draw_patches_overlay


def _attention_layers(attentions: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    if len(attentions) > 0 and torch.is_tensor(attentions[-1]) and attentions[-1].dim() <= 1:
        return attentions[:-1]
    return attentions


def _cache_positions(attentions: Sequence[torch.Tensor], seq_len: int, device: torch.device) -> torch.Tensor:
    if len(attentions) > 0 and torch.is_tensor(attentions[-1]) and attentions[-1].dim() <= 1:
        cache_pos = attentions[-1].to(device=device)
        if cache_pos.numel() == seq_len:
            return cache_pos
    return torch.arange(seq_len, device=device)


def _safe_layer_id(attentions: Sequence[torch.Tensor], layer_id: int) -> int:
    layers = _attention_layers(attentions)
    if not layers:
        raise ValueError("No attention layers available for VLA-Pruner selection.")
    return max(0, min(layer_id, len(layers) - 1))


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    score_min = scores.min()
    score_max = scores.max()
    return (scores - score_min) / (score_max - score_min + 1e-8)


def _mean_attention_to_visual_tokens(
    attentions: Sequence[torch.Tensor],
    query_start: int,
    query_end: int,
    visual_start: int,
    visual_end: int,
    layer_id: int,
) -> torch.Tensor:
    layers = _attention_layers(attentions)
    layer_id = _safe_layer_id(attentions, layer_id)
    attn_map = layers[layer_id].to(torch.float32).squeeze(0).mean(dim=0)
    cache_pos = _cache_positions(attentions, attn_map.shape[0], attn_map.device)

    query_mask = (cache_pos >= query_start) & (cache_pos < query_end)
    if query_mask.sum() == 0:
        return torch.zeros(visual_end - visual_start, dtype=torch.float32, device=attn_map.device)

    key_mask = (cache_pos >= visual_start) & (cache_pos < visual_end)
    if key_mask.numel() == attn_map.shape[1] and key_mask.sum() == visual_end - visual_start:
        return attn_map[query_mask][:, key_mask].mean(dim=0)

    visual_scores = attn_map[query_mask, visual_start:min(visual_end, attn_map.shape[1])].mean(dim=0)
    expected_len = visual_end - visual_start
    if visual_scores.numel() == expected_len:
        return visual_scores

    padded_scores = torch.zeros(expected_len, dtype=torch.float32, device=attn_map.device)
    padded_scores[: min(expected_len, visual_scores.numel())] = visual_scores[:expected_len]
    return padded_scores


def _visual_layout(metadata: Dict) -> Tuple[int, int, int, int]:
    visual_start = int(metadata.get("visual_token_start", 1))
    visual_end = int(metadata["visual_token_end"])
    patches_per_image = int(metadata.get("num_patches_per_image", 256))
    num_images = int(metadata.get("num_images", max(1, (visual_end - visual_start) // patches_per_image)))
    return visual_start, visual_end, patches_per_image, num_images


def _candidate_mask(
    metadata: Dict,
    static_token_indices: Optional[Sequence[int]],
    static_only: bool,
    device: torch.device,
) -> torch.Tensor:
    visual_start, visual_end, _, _ = _visual_layout(metadata)
    num_visual_tokens = visual_end - visual_start
    mask = torch.ones(num_visual_tokens, dtype=torch.bool, device=device)

    if static_only:
        mask.zero_()
        if static_token_indices:
            rel = torch.tensor(static_token_indices, dtype=torch.long, device=device) - visual_start
            rel = rel[(rel >= 0) & (rel < num_visual_tokens)]
            if rel.numel() > 0:
                mask[rel] = True

    return mask


def _rank_low_importance_tokens(
    scores: torch.Tensor,
    metadata: Dict,
    prune_ratio: float,
    static_token_indices: Optional[Sequence[int]],
    static_only: bool,
) -> torch.Tensor:
    visual_start, visual_end, _, _ = _visual_layout(metadata)
    num_visual_tokens = visual_end - visual_start
    if scores.numel() != num_visual_tokens:
        normalized_scores = torch.zeros(num_visual_tokens, dtype=scores.dtype, device=scores.device)
        normalized_scores[: min(num_visual_tokens, scores.numel())] = scores[:num_visual_tokens]
        scores = normalized_scores

    candidates = _candidate_mask(metadata, static_token_indices, static_only, scores.device)
    if candidates.sum() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    if static_only:
        num_prune = int(round(int(candidates.sum().item()) * prune_ratio))
    else:
        num_prune = int(round(num_visual_tokens * prune_ratio))
    num_prune = max(0, min(num_prune, int(candidates.sum().item())))
    if num_prune == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    candidate_scores = scores.clone()
    candidate_scores[~candidates] = torch.inf
    relative_indices = torch.topk(candidate_scores, num_prune, largest=False).indices
    return relative_indices + visual_start


def _make_constant_layer_schedule(attentions: Sequence[torch.Tensor], value: float = 1.0) -> torch.Tensor:
    layers = _attention_layers(attentions)
    if not layers:
        return torch.empty(0, dtype=torch.float32)
    device = layers[0].device
    return torch.full((len(layers),), float(value), dtype=torch.float32, device=device)


def compute_fastv_reuse_indices(
    attentions: Sequence[torch.Tensor],
    metadata: Dict,
    prune_ratio: float,
    layer_id: int,
    static_token_indices: Optional[Sequence[int]] = None,
    static_only: bool = False,
) -> torch.Tensor:
    """Select low-attention visual tokens using FastV's prefill attention criterion."""
    visual_start, visual_end, _, _ = _visual_layout(metadata)
    prefill_end = int(metadata.get("action_token_start", metadata["text_token_end"]))
    scores = _mean_attention_to_visual_tokens(attentions, 0, prefill_end, visual_start, visual_end, layer_id)
    scores = _normalize_scores(scores)
    return _rank_low_importance_tokens(scores, metadata, prune_ratio, static_token_indices, static_only)


def compute_semantic_action_reuse_indices(
    attentions: Sequence[torch.Tensor],
    metadata: Dict,
    prune_ratio: float,
    layer_id: int,
    semantic_weight: float,
    action_weight: float,
    static_token_indices: Optional[Sequence[int]] = None,
    static_only: bool = False,
    action_horizon: int = 0,
) -> torch.Tensor:
    """Select low-importance visual tokens using semantic and action-slot attention."""
    visual_start, visual_end, _, _ = _visual_layout(metadata)
    action_start = int(metadata["action_token_start"])
    action_end = int(metadata["action_token_end"])
    if action_horizon > 0:
        action_dim = int(metadata.get("action_dim", 7))
        action_end = min(action_end, action_start + action_horizon * action_dim)

    semantic_scores = _mean_attention_to_visual_tokens(
        attentions,
        int(metadata["text_token_start"]),
        int(metadata["text_token_end"]),
        visual_start,
        visual_end,
        layer_id,
    )
    action_scores = _mean_attention_to_visual_tokens(
        attentions,
        action_start,
        action_end,
        visual_start,
        visual_end,
        layer_id,
    )

    semantic_scores = _normalize_scores(semantic_scores)
    action_scores = _normalize_scores(action_scores)
    total_weight = max(semantic_weight + action_weight, 1e-8)
    scores = (semantic_weight * semantic_scores + action_weight * action_scores) / total_weight
    return _rank_low_importance_tokens(scores, metadata, prune_ratio, static_token_indices, static_only)


def split_visual_token_indices_by_image(token_indices: Sequence[int], metadata: Dict) -> Tuple[Sequence[int], Sequence[int]]:
    visual_start, _, patches_per_image, num_images = _visual_layout(metadata)
    primary = []
    wrist = []
    for idx in token_indices:
        rel = int(idx) - visual_start
        if rel < 0:
            continue
        image_id = rel // patches_per_image
        patch_id = rel % patches_per_image
        if image_id == 0:
            primary.append(patch_id)
        elif image_id == 1 or num_images > 1:
            wrist.append(patch_id)
    return primary, wrist


def visualize_selected_patches(images, token_indices: Sequence[int], metadata: Dict):
    primary, wrist = split_visual_token_indices_by_image(token_indices, metadata)
    result = list(images)
    if len(result) > 0 and primary:
        result[0] = draw_patches_overlay(result[0], [(primary, (40, 116, 166))], patch_size=14, alpha=0.4)
    if len(result) > 1 and wrist:
        result[1] = draw_patches_overlay(result[1], [(wrist, (40, 116, 166))], patch_size=14, alpha=0.4)
    return result


def build_vlapruner_cache_config(
    attentions: Optional[Sequence[torch.Tensor]],
    metadata: Optional[Dict],
    mode: str,
    prune_ratio: float,
    layer_id: int,
    semantic_weight: float,
    action_weight: float,
    static_token_indices: Optional[Sequence[int]] = None,
    static_only: bool = False,
    action_horizon: int = 0,
    use_layer_schedule: bool = True,
):
    if attentions is None or metadata is None:
        return None, None

    if mode == "fastv":
        reuse_indices = compute_fastv_reuse_indices(
            attentions, metadata, prune_ratio, layer_id, static_token_indices, static_only
        )
    elif mode in {"semantic_action", "vla_pruner"}:
        reuse_indices = compute_semantic_action_reuse_indices(
            attentions,
            metadata,
            prune_ratio,
            layer_id,
            semantic_weight,
            action_weight,
            static_token_indices,
            static_only,
            action_horizon,
        )
    elif mode == "semantic":
        reuse_indices = compute_semantic_action_reuse_indices(
            attentions, metadata, prune_ratio, layer_id, 1.0, 0.0, static_token_indices, static_only, action_horizon
        )
    elif mode == "action":
        reuse_indices = compute_semantic_action_reuse_indices(
            attentions, metadata, prune_ratio, layer_id, 0.0, 1.0, static_token_indices, static_only, action_horizon
        )
    else:
        raise ValueError(f"Unsupported VLA-Pruner selection mode: {mode}")

    schedule = _make_constant_layer_schedule(attentions, 1.0)
    if use_layer_schedule:
        from experiments.robot.vla_cache_utils import get_layer_mask_schedule

        schedule = get_layer_mask_schedule(attentions)

    return reuse_indices, schedule
