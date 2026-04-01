import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _ensure_results(df: pd.DataFrame) -> pd.DataFrame:
    req = {"w_time", "w_profit", "seed", "fitness", "time", "profit"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df

def _aggregate(df: pd.DataFrame, agg_mode: str = "mean") -> pd.DataFrame:
    df = df.copy()
    df["w_time"] = df["w_time"].astype(float)
    df["w_profit"] = df["w_profit"].astype(float)

    if agg_mode == "best":
        idx = df.groupby(["w_time", "w_profit"])["fitness"].idxmin()
        best = df.loc[idx.to_numpy()].sort_values("w_time").reset_index(drop=True)
        return best.rename(columns={
            "fitness": "fitness_best",
            "time": "time_best",
            "profit": "profit_best"
        })

    g = df.groupby(["w_time", "w_profit"], as_index=False).agg(
        fitness_mean=("fitness", "mean"),
        fitness_std=("fitness", "std"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
        profit_mean=("profit", "mean"),
        profit_std=("profit", "std"),
        n=("seed", "nunique")
    ).sort_values("w_time").reset_index(drop=True)
    return g

def _plot_series(x, y, yerr, xlabel, ylabel, title, out_png):
    plt.figure(figsize=(8, 5))
    if yerr is not None and not np.isnan(yerr).all():
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    else:
        plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

def plot_weights(csv_path: Path, out_prefix: Path, agg_mode: str = "mean"):
    df = pd.read_csv(csv_path)
    df = _ensure_results(df)

    agg = _aggregate(df, agg_mode=agg_mode)
    instance = df["instance"].iloc[0] if "instance" in df.columns and len(df) else csv_path.name
    label = "(mean±std)" if agg_mode == "mean" else "(best)"

    if agg_mode == "mean":
        y, yerr = agg["fitness_mean"], agg["fitness_std"]
    else:
        y, yerr = agg["fitness_best"], None
    _plot_series(
        agg["w_time"], y, yerr,
        xlabel="Weight on TIME (w_time)", ylabel="Fitness (minimise)",
        title=f"Scalarised fitness vs w_time — {instance} {label}",
        out_png=f"{out_prefix}_fitness.png"
    )

    if agg_mode == "mean":
        y, yerr = agg["time_mean"], agg["time_std"]
    else:
        y, yerr = agg["time_best"], None
    _plot_series(
        agg["w_time"], y, yerr,
        xlabel="Weight on TIME (w_time)", ylabel="Travel time (minimise)",
        title=f"Travel time vs w_time — {instance} {label}",
        out_png=f"{out_prefix}_time.png"
    )

    if agg_mode == "mean":
        y, yerr = agg["profit_mean"], agg["profit_std"]
    else:
        y, yerr = agg["profit_best"], None
    _plot_series(
        agg["w_time"], y, yerr,
        xlabel="Weight on TIME (w_time)", ylabel="Profit (maximise)",
        title=f"Profit vs w_time — {instance} {label}",
        out_png=f"{out_prefix}_profit.png"
    )

def main():
    ap = argparse.ArgumentParser(description="Plot weight sensitivity for single-objective runs.")
    ap.add_argument("--csv", default="results/results_single.csv",
                    help="Path to results_single.csv (e.g., results/a280_results_single.csv).")
    ap.add_argument("--out_prefix", default="results/weights_sensitivity",
                    help="Prefix for output files (no extension).")
    ap.add_argument("--mode", choices=["mean", "best"], default="mean",
                    help="Aggregate across seeds with mean±std or pick best per weight.")
    args = ap.parse_args()

    plot_weights(Path(args.csv), Path(args.out_prefix), agg_mode=args.mode)

if __name__ == "__main__":
    main()