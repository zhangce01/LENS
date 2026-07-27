import numpy as np
import torch
import torch.nn.functional as F


def uniform_sampling(raw_video, question, num_frames):
    T = len(raw_video)
    idx = np.linspace(0, T - 1, num_frames)
    idx = np.round(idx).astype(np.int64)
    idx = np.clip(idx, 0, T - 1)
    selected_indices = idx.tolist()
    return selected_indices


def _get_clip_features_and_scores(clip_model, clip_processor, raw_video, question, device, batch_size=64):
    """
    Encode all frames with CLIP and compute text-image similarity scores.
    Returns:
        V: (T, D) normalized image features
        scores: (T,) raw CLIP similarity (text @ image_features)
    """
    T = len(raw_video)
    all_features = []
    with torch.no_grad():
        inputs_text = clip_processor(
            text=[question],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_features = clip_model.get_text_features(**inputs_text)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)  # (1, D)

    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch_images = raw_video[i : i + batch_size].to(device)
            inputs_image = clip_processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
            ).to(device)
            image_features = clip_model.get_image_features(**inputs_image)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            all_features.append(image_features)

    V = torch.cat(all_features, dim=0)  # (T, D)
    scores = (text_features @ V.T).squeeze(0)  # (T,)
    return V, scores


def _get_blip2_features_and_scores(blip2_model, blip2_processor, raw_video, question, device, batch_size=64):
    """
    Encode all frames with BLIP2 ITM and compute text-image matching scores.
    Returns:
        V: None (BLIP2 ITM does not expose normalized image features for graph)
        scores: (T,) ITM match probability (positive class) per frame
    """
    T = len(raw_video)
    do_rescale = raw_video.max() > 1.0 if hasattr(raw_video, 'max') else True
    all_scores = []
    with torch.no_grad():
        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            batch_frames = [raw_video[j] for j in range(i, end)]
            if isinstance(batch_frames[0], torch.Tensor):
                batch_frames = [f.cpu() if f.device.type != "cpu" else f for f in batch_frames]
            inputs = blip2_processor(
                images=batch_frames,
                text=[question] * len(batch_frames),
                return_tensors="pt",
                do_rescale=do_rescale,
            ).to(device)
            outputs = blip2_model(**inputs, use_image_text_matching_head=True)
            logits = outputs.logits_per_image  # (batch, 2)
            probs = torch.softmax(logits, dim=1)
            match_scores = probs[:, 1]  # positive class
            all_scores.append(match_scores)
    scores = torch.cat(all_scores, dim=0)  # (T,)
    return None, scores


def _get_blip_features_and_scores(blip_model, blip_processor, raw_video, question, device, batch_size=64):
    """
    Encode all frames with BLIP ITM using Hugging Face transformers.
    """
    T = len(raw_video)
    all_scores = []

    blip_model.eval()

    with torch.no_grad():
        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            batch_frames = [raw_video[j] for j in range(i, end)]

            # The processor prepares both image and text tensors
            inputs = blip_processor(
                images=batch_frames,
                text=[question] * len(batch_frames),
                padding=True,
                return_tensors="pt"
            ).to(device)

            outputs = blip_model(**inputs)

            # HF returns 'itm_score': logits for [unmatch, match], shape (batch, 2)
            logits = outputs.itm_score
            probs = torch.softmax(logits, dim=-1)

            # Index 1 is the match probability
            match_scores = probs[:, 1]
            all_scores.append(match_scores)

    scores = torch.cat(all_scores, dim=0)
    return None, scores


def _compute_ssim_matrix(frames: torch.Tensor, device: torch.device, max_size: int = 224) -> torch.Tensor:
    """Compute the pairwise SSIM matrix over all frames, fully vectorized in PyTorch."""
    T, C, H, W = frames.shape
    # 1. Downsample for speed
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
        frames = F.interpolate(frames.float(), size=(int(H*scale), int(W*scale)), mode="area")

    frames = frames.to(device).float()
    data_range = 255.0 if frames.max() > 1.0 else 1.0

    # 2. SSIM window (Gaussian kernel)
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g.view(1, 1, window_size, 1) * g.view(1, 1, 1, window_size)
    kernel = kernel.expand(C, 1, window_size, window_size).contiguous()

    # 3. Per-frame local statistics (parallel over T): mean and variance
    mu = F.conv2d(frames, kernel, padding=window_size//2, groups=C)
    mu_sq = mu.pow(2)
    # sigma^2 = E[X^2] - (E[X])^2
    sigma_sq = F.conv2d(frames * frames, kernel, padding=window_size//2, groups=C) - mu_sq

    # 4. Pairwise covariances -> SSIM matrix A[i, j]
    A = torch.zeros((T, T), device=device)
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2

    # One frame vs. all frames per iteration (row-wise parallel) to avoid OOM
    for i in range(T):
        mu_i = mu[i:i+1]           # (1, C, h, w)
        sigma_sq_i = sigma_sq[i:i+1]
        frame_i = frames[i:i+1]

        # Covariance: sigma_xy = E[XY] - E[X]E[Y]
        # frame_i * frames broadcasts: (1,C,h,w) * (T,C,h,w) -> (T,C,h,w)
        mu_ij = mu_i * mu
        sigma_xy = F.conv2d(frame_i * frames, kernel, padding=window_size//2, groups=C) - mu_ij

        # SSIM: ((2*mu_i*mu_j + C1) * (2*sigma_xy + C2)) / ((mu_i^2 + mu_j^2 + C1) * (sigma_i^2 + sigma_j^2 + C2))
        num = (2 * mu_ij + C1) * (2 * sigma_xy + C2)
        den = (mu_sq[i:i+1] + mu_sq + C1) * (sigma_sq_i + sigma_sq + C2)

        ssim_map = num / den
        # Average over spatial dims -> (T,) vector
        A[i] = ssim_map.mean(dim=(1, 2, 3))

    # Map from [-1, 1] to [0, 1]
    return (A + 1.0) / 2.0


def normalized_kendall_tau(rank1, rank2):
    """
    Calculates the Normalized Kendall's Tau between two rankings.
    rank1, rank2: 1D Tensors containing the rank/value of items.
    Returns: A float between -1 (opposite) and 1 (identical).
    """
    n = rank1.size(0)
    if n < 2:
        return 1.0

    # All pairwise combinations (i, j) with i < j
    i, j = torch.triu_indices(n, n, offset=1)

    # Relative ordering in each ranking
    order1 = rank1[i] < rank1[j]
    order2 = rank2[i] < rank2[j]

    # Concordant pairs agree on the order, discordant pairs disagree
    concordant = (order1 == order2).sum().item()
    discordant = (order1 != order2).sum().item()

    # Normalized tau: (C - D) / total pairs
    total_pairs = n * (n - 1) // 2
    tau = (concordant - discordant) / total_pairs

    return tau


def _graph_diffusion_refine(
    y: torch.Tensor,
    A_sem: torch.Tensor,
    beta: float = 0.8,
    num_iters: int = 5,
    sigma: float = 2.0,
):
    """
    Graph-based message passing (manifold ranking) to refine initial scores y.
    y: (T,) initial ITM similarity scores
    A_sem: (T, T) precomputed inter-frame similarity (e.g. SSIM or CLIP). Connect the most similar
           frames according to pairwise similarity until all frames are connected.
    Returns: refined scores f, plus diagnostics (confidence shift, Kendall tau, normalized entropy).
    """

    # 1. Edge list sorted by similarity (descending)
    T, device = A_sem.shape[0], A_sem.device
    triu_indices = torch.triu_indices(T, T, offset=1, device=device)
    similarities = A_sem[triu_indices[0], triu_indices[1]]
    sorted_indices = torch.argsort(similarities, descending=True)
    sorted_edges = triu_indices[:, sorted_indices]

    # 2. Initialize adjacency W and union-find
    W = torch.zeros((T, T), device=device)
    parent = torch.arange(T, device=device)
    num_components = T  # every node starts as its own component

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    # 3. Add edges from most to least similar
    for k in range(sorted_edges.shape[1]):
        u, v = sorted_edges[0, k].item(), sorted_edges[1, k].item()

        # Every edge taken (cycle or not) is added to W with a Gaussian-kernel weight
        sim = A_sem[u, v]
        dist_sq = 2 * (1.0 - sim)
        weight = torch.exp(-dist_sq / (2 * (sigma**2)))
        W[u, v] = W[v, u] = weight

        # Track connectivity with union-find
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            num_components -= 1

        # Stop once the graph is fully connected
        if num_components == 1:
            break

    # Symmetrically normalized transition: P = D^{-1/2} W D^{-1/2}
    deg = W.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    P = deg_inv_sqrt.unsqueeze(1) * W * deg_inv_sqrt.unsqueeze(0)

    # Diffusion: f_{t+1} = beta * P @ f_t + (1 - beta) * y
    f = y.float().clone()
    for _ in range(num_iters):
        f = beta * (P @ f) + (1 - beta) * y.float()

    # Confidence shift between refined scores f and initial scores y
    confidence_shift = F.cosine_similarity(f.unsqueeze(0), y.unsqueeze(0), dim=1)

    rank1 = torch.argsort(f, descending=True)
    rank2 = torch.argsort(y, descending=True)
    tau = normalized_kendall_tau(rank1, rank2)

    temp = 0.05
    probs = torch.softmax(f / temp, dim=0)

    # Shannon entropy H = -sum(p * log(p)); 1e-9 avoids log(0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9))

    # Normalize by log(T) so the result is in [0, 1]:
    # 0 = point distribution (one frame has all the weight)
    # 1 = uniform distribution (all frames equally weighted)
    max_entropy = torch.log(torch.tensor(T, dtype=torch.float, device=device))
    norm_entropy = entropy / max_entropy

    return f, confidence_shift, tau, norm_entropy


def _compute_clip_feature_similarity_matrix(clip_model, clip_processor, raw_video, question, num_frames, batch_size=64):
    device = next(clip_model.parameters()).device
    T = len(raw_video)
    V, _ = _get_clip_features_and_scores(clip_model, clip_processor, raw_video, question, device, batch_size)
    return V @ V.T


def clip_based_sampling(clip_model, clip_processor, raw_video, question, num_frames, batch_size=64):
    device = next(clip_model.parameters()).device
    T = len(raw_video)
    _, scores = _get_clip_features_and_scores(clip_model, clip_processor, raw_video, question, device, batch_size)
    selected_indices = select_anchor_frames_watershed(scores, K_anchor=num_frames)
    if len(selected_indices) < num_frames:
        selected_indices = torch.topk(scores, k=min(num_frames, T)).indices
        selected_indices = torch.sort(selected_indices).values.tolist()
    return selected_indices, scores


def blip2_based_sampling(blip2_model, blip2_processor, raw_video, question, num_frames, batch_size=64):
    device = next(blip2_model.parameters()).device
    T = len(raw_video)
    _, scores = _get_blip2_features_and_scores(blip2_model, blip2_processor, raw_video, question, device, batch_size)
    selected_indices = select_anchor_frames_watershed(scores, K_anchor=num_frames)
    if len(selected_indices) < num_frames:
        selected_indices = torch.topk(scores, k=min(num_frames, T)).indices
        selected_indices = torch.sort(selected_indices).values.tolist()
    return selected_indices, scores


def blip_based_sampling(blip_model, blip_processor, raw_video, question, num_frames, batch_size=64):
    device = next(blip_model.parameters()).device
    T = len(raw_video)
    _, scores = _get_blip_features_and_scores(blip_model, blip_processor, raw_video, question, device, batch_size)
    selected_indices = select_anchor_frames_watershed(scores, K_anchor=num_frames)
    if len(selected_indices) < num_frames:
        selected_indices = torch.topk(scores, k=min(num_frames, T)).indices
        selected_indices = torch.sort(selected_indices).values.tolist()
    return selected_indices, scores


def graph_based_sampling(
    raw_video, scores, num_frames, device, batch_size=64,
    beta=0.8, num_iters=10, sigma=2.0, use_ssim=True, clip_model=None, clip_processor=None, question=None
):
    """
    Graph-based message passing (manifold ranking) for score refinement,
    then watershed anchor selection. Uses same fallback as clip_based_sampling when too few anchors.
    use_ssim: if True, use SSIM (structural similarity) for the inter-frame semantic matrix
              instead of CLIP feature similarity.
    """
    T = len(raw_video)
    # raw_video: (T, C, H, W)

    if use_ssim:
        A_sem = _compute_ssim_matrix(raw_video, device)
    else:
        A_sem = _compute_clip_feature_similarity_matrix(clip_model, clip_processor, raw_video, question, num_frames, batch_size)
    refined_scores, confidence_shift, tau, norm_entropy = _graph_diffusion_refine(
        scores, A_sem=A_sem,
        beta=beta, num_iters=num_iters, sigma=sigma,
    )
    selected_indices = select_anchor_frames_watershed(refined_scores, K_anchor=num_frames)
    if len(selected_indices) < num_frames:
        selected_indices = torch.topk(refined_scores, k=min(num_frames, T)).indices
        selected_indices = torch.sort(selected_indices).values.tolist()
    return selected_indices, refined_scores, confidence_shift, tau, norm_entropy


def _find_valleys(scores: torch.Tensor) -> torch.Tensor:
    s = scores
    T = s.numel()
    if T == 0:
        return torch.empty(0, dtype=torch.long, device=s.device)
    if T == 1:
        return torch.tensor([0], dtype=torch.long, device=s.device)

    left  = s[1:-1] <= s[:-2]
    right = s[1:-1] <  s[2:]
    left2 = s[1:-1] <  s[:-2]
    right2= s[1:-1] <= s[2:]
    interior = (left & right) | (left2 & right2)
    valleys = torch.nonzero(interior, as_tuple=False).squeeze(-1) + 1  # shift for 1:-1

    # endpoints
    if s[0] <= s[1]:
        valleys = torch.cat([torch.tensor([0], device=s.device), valleys])
    if s[-1] <= s[-2]:
        valleys = torch.cat([valleys, torch.tensor([T - 1], device=s.device)])

    # unique + sorted
    valleys = torch.unique(valleys, sorted=True)
    return valleys


def _kmeans_1d(x: torch.Tensor, k: int, iters: int = 30) -> torch.Tensor:
    """
    Simple 1D k-means on x (N,) -> assignments (N,)
    x should be float tensor.
    """
    N = x.numel()
    if k <= 1 or N == 0:
        return torch.zeros(N, dtype=torch.long, device=x.device)
    if k >= N:
        return torch.arange(N, dtype=torch.long, device=x.device)

    # init centers by picking evenly spaced points in sorted x
    xs, order = torch.sort(x)
    init_ids = torch.linspace(0, N - 1, steps=k, device=x.device).round().long()
    centers = xs[init_ids].clone()

    for _ in range(iters):
        # assign
        d2 = (x[:, None] - centers[None, :]).abs()  # (N, k)
        assign = torch.argmin(d2, dim=1)

        # update
        new_centers = centers.clone()
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                new_centers[j] = x[mask].mean()
            else:
                # empty cluster: re-seed to a random point
                new_centers[j] = x[torch.randint(0, N, (1,), device=x.device)].item()

        if torch.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    return assign


def select_anchor_frames_watershed(scores: torch.Tensor, K_anchor: int):
    """
    scores: (T,) similarity curve
    returns: sorted python list of selected frame indices
    """
    assert scores.dim() == 1, "scores must be 1D (T,)"
    T = scores.numel()
    if T == 0 or K_anchor <= 0:
        return []

    # 1) find valleys -> basin boundaries
    valleys = _find_valleys(scores)

    # ensure boundaries include 0 and T-1
    boundaries = valleys
    if boundaries.numel() == 0:
        boundaries = torch.tensor([0, T - 1], device=scores.device, dtype=torch.long)
    else:
        if boundaries[0].item() != 0:
            boundaries = torch.cat([torch.tensor([0], device=scores.device), boundaries])
        if boundaries[-1].item() != T - 1:
            boundaries = torch.cat([boundaries, torch.tensor([T - 1], device=scores.device)])
        boundaries = torch.unique(boundaries, sorted=True)

    # 2) for each basin (between consecutive valleys), pick peak as candidate
    candidates = []
    for i in range(boundaries.numel() - 1):
        start = boundaries[i].item()
        end   = boundaries[i + 1].item()
        if end < start:
            continue
        seg = scores[start : end + 1]
        if seg.numel() == 0:
            continue
        peak_local = torch.argmax(seg).item()
        candidates.append(start + peak_local)

    # dedup candidates (can happen with tiny basins)
    if len(candidates) == 0:
        # fallback: global top1
        return [int(torch.argmax(scores).item())]

    cand = torch.tensor(sorted(set(candidates)), device=scores.device, dtype=torch.long)

    # 3) if too many candidates, cluster by temporal index into K_anchor groups
    if cand.numel() > K_anchor:
        x = cand.float()
        assign = _kmeans_1d(x, K_anchor, iters=30)  # (Nc,)

        selected = []
        for j in range(K_anchor):
            mask = (assign == j)
            if not mask.any():
                continue
            idxs = cand[mask]                  # candidate frame indices
            vals = scores[idxs]                # their scores
            best = idxs[torch.argmax(vals)].item()
            selected.append(best)

        selected = sorted(set(selected))
        return selected[:K_anchor]  # safety
    else:
        # 4) else: use all candidates (up to K_anchor)
        return cand[:K_anchor].sort().values.tolist()
