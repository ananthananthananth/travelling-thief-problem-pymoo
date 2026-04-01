import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def _ensure_profit_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "neg_profit" in df.columns and "profit" not in df.columns:
        df = df.copy()
        df["profit"] = -df["neg_profit"]
    return df

def _non_dominated_mask_time_profit(df: pd.DataFrame) -> np.ndarray:
    T = df["time"].to_numpy()
    P = df["profit"].to_numpy()
    n = len(df)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        ti, pi = T[i], P[i]
        dominated = (T <= ti) & (P >= pi) & ((T < ti) | (P > pi))
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return keep

def plot_pareto_fronts(combined_csv: Path, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if combined_csv.exists():
        df = pd.read_csv(combined_csv)
    else:
        base = combined_csv.parent if combined_csv.parent.as_posix() != "" else Path("results")
        frames = []
        for algo in ["nsga2", "moead", "nsga3"]:
            fn = base / f"front_{algo}.csv"
            if fn.exists():
                d = pd.read_csv(fn)
                d["algo"] = algo
                frames.append(d)
        if not frames:
            raise FileNotFoundError("No combined CSV or front_{algo}.csv files found in results/.")
        df = pd.concat(frames, ignore_index=True)

    df = _ensure_profit_columns(df)

    instance = df["instance"].iloc[0] if "instance" in df.columns and len(df) else "unknown instance"
    gens = df["gens"].iloc[0] if "gens" in df.columns and len(df) else None
    pop = df["pop"].iloc[0] if "pop" in df.columns and len(df) else None
    subtitle = "  ".join([s for s in [f"gens={gens}" if gens is not None else None,
                                      f"pop={pop}" if pop is not None else None] if s])

    plt.figure(figsize=(8, 6))
    if "algo" in df.columns:
        for algo, g in df.groupby("algo"):
            g = _ensure_profit_columns(g)
            plt.scatter(g["time"], g["profit"], s=18, alpha=0.7, label=algo)
    else:
        plt.scatter(df["time"], df["profit"], s=18, alpha=0.7, label="front")

    mask = _non_dominated_mask_time_profit(df)
    nd = df[mask].sort_values("time")
    plt.plot(nd["time"], nd["profit"], linewidth=2, label="union ND curve")

    title = f"Pareto fronts: {instance}"
    if subtitle:
        title += f"\n{subtitle}"
    plt.title(title)
    plt.xlabel("Travel time (minimise)")
    plt.ylabel("Profit (maximise)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

def main():
    ap = argparse.ArgumentParser(description="Plot Pareto fronts for TTP multi objective runs.")
    ap.add_argument("--combined", default="results/fronts_combined.csv", help="Combined CSV from run_multi")
    ap.add_argument("--out_png", default="results/pareto_fronts.png", help="Output PNG path")
    args = ap.parse_args()

    plot_pareto_fronts(Path(args.combined), Path(args.out_png))

if __name__ == "__main__":
    main()
