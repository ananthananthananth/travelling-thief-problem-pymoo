import argparse
import pandas as pd
from pathlib import Path
from .ttp_pymoo import run_ga_single, run_de_single

def main():
    ap = argparse.ArgumentParser(description="Single objective TTP runs (GA / DE) with weighted scalarization.")
    ap.add_argument("--inst", default="data/a280-n279.txt", help="Path to TTP instance file")
    ap.add_argument("--gens", type=int, default=200, help="Number of generations")
    ap.add_argument("--pop", type=int, default=120, help="Population size")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds")
    ap.add_argument("--weights", type=float, nargs="*", default=[1.0, 0.75, 0.5, 0.25, 0.0],
                    help="Weights for time in [0,1]. Profit weight is 1-w_time.")
    ap.add_argument("--algo", choices=["ga", "de", "both"], default="ga",
                    help="Which single-objective algorithm to run.")
    ap.add_argument("--prefix", default="", help="Prefix for output filenames, e.g., a280_ or fnl_")
    ap.add_argument("--outdir", default="results", help="Output directory")
    # DE hyper-params (optional tweaks)
    ap.add_argument("--deF", type=float, default=0.5, help="DE mutation factor F")
    ap.add_argument("--deCR", type=float, default=0.9, help="DE crossover rate CR")
    ap.add_argument("--deVariant", default="DE/rand/1/bin", help="DE variant, e.g., 'DE/rand/1/bin'")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def run_and_collect(runner_name: str):
        rows = []
        for w_time in args.weights:
            w_profit = 1.0 - w_time
            for s in args.seeds:
                if runner_name == "ga":
                    res = run_ga_single(args.inst, w_time, w_profit,
                                        n_gen=args.gens, pop_size=args.pop, seed=s)
                else:
                    res = run_de_single(args.inst, w_time, w_profit,
                                        n_gen=args.gens, pop_size=args.pop, seed=s,
                                        F=args.deF, CR=args.deCR, variant=args.deVariant)
                rows.append({
                    "algo": runner_name.upper(),
                    "instance": Path(args.inst).name,
                    "w_time": w_time,
                    "w_profit": w_profit,
                    "seed": s,
                    "fitness": res["best_f"],
                    "time": res["time"],
                    "profit": res["profit"],
                    "net": res["net"],
                    "gens": args.gens,
                    "pop": args.pop
                })
        return pd.DataFrame(rows)

    if args.algo in ("ga", "both"):
        df_ga = run_and_collect("ga")
        ga_path = outdir / f"{args.prefix}results_single_ga.csv"
        df_ga.to_csv(ga_path, index=False)
        print(f"Wrote {ga_path.as_posix()}")
        print(df_ga.head())

    if args.algo in ("de", "both"):
        df_de = run_and_collect("de")
        de_path = outdir / f"{args.prefix}results_single_de.csv"
        df_de.to_csv(de_path, index=False)
        print(f"Wrote {de_path.as_posix()}")
        print(df_de.head())

if __name__ == "__main__":
    main()