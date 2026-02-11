from __future__ import annotations

import glob
import random

import pandas as pd
import torch

IN_DIM = 48
HIDDEN_DIM = 96
OUT_DIM = 1
CSV_PATH = "historical_data_and_pieces/historical_data.csv"
PIECES_GLOB = "historical_data_and_pieces/pieces/piece_*.pth"
SWAP_ITERS = 10000
RANDOM_SEED = 3
TARGET_RMSE = 1e-6
LOG_EVERY = 500


def rmse(preds: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((preds - ys) ** 2))


def load_problem_data(csv_path: str, pieces_glob: str):
    df = pd.read_csv(csv_path)
    X = torch.tensor(
        df[[f"measurement_{i}" for i in range(IN_DIM)]].values, dtype=torch.float32
    )
    y_pred = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)
    y_true = torch.tensor(df["true"].values, dtype=torch.float32).unsqueeze(1)

    in_pieces = []
    out_pieces = []
    last_piece = None

    for piece_path in sorted(
        glob.glob(pieces_glob),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    ):
        piece_id = int(piece_path.split("_")[-1].split(".")[0])
        state = torch.load(piece_path, map_location="cpu")
        weight = state["weight"].float()
        bias = state["bias"].float()
        shape = tuple(weight.shape)

        if shape == (HIDDEN_DIM, IN_DIM):
            in_pieces.append((piece_id, weight, bias))
        elif shape == (IN_DIM, HIDDEN_DIM):
            out_pieces.append((piece_id, weight, bias))
        elif shape == (OUT_DIM, IN_DIM):
            last_piece = (piece_id, weight, bias)
        else:
            raise ValueError(f"Unexpected piece shape {shape} for piece_{piece_id}")

    if len(in_pieces) != 48 or len(out_pieces) != 48 or last_piece is None:
        raise ValueError("Expected 48 input pieces, 48 output pieces, and 1 last layer")

    return X, y_pred, y_true, in_pieces, out_pieces, last_piece


def cosine_similarity_with_X(X: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    x_flat = X.reshape(-1)
    r_flat = residual.reshape(-1)
    return torch.dot(x_flat, r_flat) / (
        torch.norm(x_flat) * torch.norm(r_flat) + 1e-12
    )


def build_blocks_from_abs_cosine(X: torch.Tensor, in_pieces, out_pieces):
    """Step 1: score all 48x48 pairs, then pick 48 one-to-one pairs greedily by |cos|."""
    scored_pairs = []
    for in_idx, (in_id, w1, b1) in enumerate(in_pieces):
        hidden = torch.relu(X @ w1.T + b1)
        for out_idx, (out_id, w2, b2) in enumerate(out_pieces):
            residual = hidden @ w2.T + b2
            cos = cosine_similarity_with_X(X, residual).item()
            scored_pairs.append((abs(cos), cos, in_idx, out_idx, in_id, out_id))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    used_in = set()
    used_out = set()
    blocks = []
    for abs_cos, cos, in_idx, out_idx, in_id, out_id in scored_pairs:
        if in_idx in used_in or out_idx in used_out:
            continue
        blocks.append(
            {
                "in_idx": in_idx,
                "out_idx": out_idx,
                "in_id": in_id,
                "out_id": out_id,
                "abs_cos": abs_cos,
                "cos": cos,
            }
        )
        used_in.add(in_idx)
        used_out.add(out_idx)
        if len(blocks) == 48:
            break

    if len(blocks) != 48:
        raise RuntimeError("Failed to build 48 unique blocks from cosine scores")
    return blocks


def residual_l2_norm(X: torch.Tensor, in_piece, out_piece) -> float:
    _, w1, b1 = in_piece
    _, w2, b2 = out_piece
    hidden = torch.relu(X @ w1.T + b1)
    residual = hidden @ w2.T + b2
    return torch.norm(residual, dim=1).mean().item()


def model_preds_from_blocks(X: torch.Tensor, blocks, in_pieces, out_pieces, last_piece):
    h = X
    for b in blocks:
        _, w1, b1 = in_pieces[b["in_idx"]]
        _, w2, b2 = out_pieces[b["out_idx"]]
        hidden = torch.relu(h @ w1.T + b1)
        h = h + (hidden @ w2.T + b2)
    _, w_last, b_last = last_piece
    return h @ w_last.T + b_last


def optimize_block_order(
    X: torch.Tensor,
    y_pred: torch.Tensor,
    blocks,
    in_pieces,
    out_pieces,
    last_piece,
    iters: int,
    seed: int,
    target_rmse: float,
    log_every: int,
):
    """Step 3: swap fixed blocks to minimize RMSE against `pred`."""
    random.seed(seed)

    current = list(blocks)
    with torch.no_grad():
        current_rmse = rmse(
            model_preds_from_blocks(X, current, in_pieces, out_pieces, last_piece), y_pred
        ).item()
    best = list(current)
    best_rmse = current_rmse

    for it in range(1, iters + 1):
        i, j = random.sample(range(len(current)), 2)
        current[i], current[j] = current[j], current[i]

        with torch.no_grad():
            cand_rmse = rmse(
                model_preds_from_blocks(X, current, in_pieces, out_pieces, last_piece),
                y_pred,
            ).item()

        if cand_rmse <= current_rmse:
            current_rmse = cand_rmse
            if cand_rmse < best_rmse:
                best_rmse = cand_rmse
                best = list(current)
        else:
            current[i], current[j] = current[j], current[i]

        if log_every > 0 and it % log_every == 0:
            print(f"iter={it} current_rmse={current_rmse:.12f} best_rmse={best_rmse:.12f}")
        if best_rmse <= target_rmse:
            print(f"Reached target RMSE {best_rmse:.12f} at iter {it}")
            break

    return best, best_rmse


if __name__ == "__main__":
    X, y_pred, y_true, in_pieces, out_pieces, last_piece = load_problem_data(
        CSV_PATH, PIECES_GLOB
    )

    # 1) Build blocks from absolute cosine similarity.
    blocks = build_blocks_from_abs_cosine(X, in_pieces, out_pieces)

    # 2) Sort blocks by residual L2 norm (lowest -> highest).
    for b in blocks:
        b["residual_l2"] = residual_l2_norm(
            X, in_pieces[b["in_idx"]], out_pieces[b["out_idx"]]
        )
    blocks.sort(key=lambda b: b["residual_l2"])

    with torch.no_grad():
        baseline_preds = model_preds_from_blocks(X, blocks, in_pieces, out_pieces, last_piece)
        baseline_pred_rmse = rmse(baseline_preds, y_pred).item()
        baseline_true_rmse = rmse(baseline_preds, y_true).item()
    print(
        f"Baseline after L2 sort: rmse_vs_pred={baseline_pred_rmse:.12f} "
        f"rmse_vs_true={baseline_true_rmse:.12f}"
    )

    # 3) Swap block pairs until RMSE against pred approaches zero.
    best_blocks, best_pred_rmse = optimize_block_order(
        X=X,
        y_pred=y_pred,
        blocks=blocks,
        in_pieces=in_pieces,
        out_pieces=out_pieces,
        last_piece=last_piece,
        iters=SWAP_ITERS,
        seed=RANDOM_SEED,
        target_rmse=TARGET_RMSE,
        log_every=LOG_EVERY,
    )

    with torch.no_grad():
        best_preds = model_preds_from_blocks(
            X, best_blocks, in_pieces, out_pieces, last_piece
        )
        best_true_rmse = rmse(best_preds, y_true).item()

    permutation = []
    for b in best_blocks:
        permutation.append(b["in_id"])
        permutation.append(b["out_id"])
    permutation.append(last_piece[0])

    print(f"Final rmse_vs_pred={best_pred_rmse:.12f}")
    print(f"Final rmse_vs_true={best_true_rmse:.12f}")
    print("Final permutation:")
    print(permutation)
