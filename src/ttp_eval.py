# src/ttp_eval.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume

def _spacing(F: np.ndarray) -> float:
    if len(F) <= 1:
        return 0.0
    from scipy.spatial import cKDTree
    tree = cKDTree(F)
    dists, _ = tree.query(F, k=2) 
    d = dists[:, 1]
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))


def _eval_front(df_sub: pd.DataFrame, ref: np.ndarray, ideal: np.ndarray, box_volume: float) -> dict:
    F = df_sub[["time", "neg_profit"]].to_numpy(dtype=float)
    hv_raw = Hypervolume(ref_point=ref).do(F)
    sp = _spacing(F)

    hv_norm = float(hv_raw / box_volume) if box_volume > 0 else np.nan

    return {
        "hypervolume": float(hv_raw),
        "hypervolume_norm": hv_norm,
        "spacing": float(sp),
        "time_range": float(df_sub["time"].max() - df_sub["time"].min()),
        "profit_range": float((-df_sub["neg_profit"]).max() - (-df_sub["neg_profit"]).min()),
        "n_points": int(len(df_sub)),
    }

def evaluate(combined_csv: Path, out_prefix: Path):
    df = pd.read_csv(combined_csv)

    ref_time = float(1.05 * df["time"].max())
    ref_np   = float(1.05 * df["neg_profit"].max())
    ref = np.array([ref_time, ref_np], dtype=float)

    ideal_time = float(df["time"].min())
    ideal_np   = float(df["neg_profit"].min())
    ideal = np.array([ideal_time, ideal_np], dtype=float)

    box_volume = (ref_time - ideal_time) * (ref_np - ideal_np)
    box_volume = float(box_volume)

    has_seed = "seed" in df.columns
    group_cols = ["algo", "seed"] if has_seed else ["algo"]

    rows = []
    for keys, g in df.groupby(group_cols):
        rec = {k: v for k, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}
        rec.update(_eval_front(g, ref=ref, ideal=ideal, box_volume=box_volume))
        rows.append(rec)

    per_seed = pd.DataFrame(rows)
    per_seed["ref_time"] = ref_time
    per_seed["ref_neg_profit"] = ref_np
    per_seed["ideal_time"] = ideal_time
    per_seed["ideal_neg_profit"] = ideal_np
    per_seed["box_volume"] = box_volume

    out_per_seed = f"{out_prefix}_per_seed.csv" if has_seed else f"{out_prefix}"
    Path(out_per_seed).parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_per_seed, index=False)
    print(f"Wrote {out_per_seed}")

    if has_seed:
        agg = per_seed.groupby("algo").agg(
            hv_mean=("hypervolume", "mean"),
            hv_std=("hypervolume", "std"),
            hv_norm_mean=("hypervolume_norm", "mean"),
            hv_norm_std=("hypervolume_norm", "std"),
            spacing_mean=("spacing", "mean"),
            spacing_std=("spacing", "std"),
            time_range_mean=("time_range", "mean"),
            time_range_std=("time_range", "std"),
            profit_range_mean=("profit_range", "mean"),
            profit_range_std=("profit_range", "std"),
            n_runs=("seed", "nunique"),
        ).reset_index()

        agg["ref_time"] = ref_time
        agg["ref_neg_profit"] = ref_np
        agg["ideal_time"] = ideal_time
        agg["ideal_neg_profit"] = ideal_np
        agg["box_volume"] = box_volume

        out_summary = f"{out_prefix}_summary.csv"
        agg.to_csv(out_summary, index=False)
        print(f"Wrote {out_summary}")

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate MO fronts: hypervolume (raw & normalised), spacing, ranges."
    )
    ap.add_argument(
        "--combined",
        required=True,
        help="Path to fronts_combined.csv (single-seed) or fronts_combined_all_seeds.csv (with 'seed' column).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output prefix. If input has seeds, writes *_per_seed.csv and *_summary.csv; "
             "otherwise writes the single summary CSV directly.",
    )
    args = ap.parse_args()

    evaluate(Path(args.combined), Path(args.out))

if __name__ == "__main__":
    main()