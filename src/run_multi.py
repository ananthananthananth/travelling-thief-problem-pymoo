# src/run_multi.py
import argparse
from pathlib import Path
import pandas as pd
from .ttp_pymoo import run_nsga2, run_nsga3, run_moead

def _df_from_res(res, algo_name: str, instance_name: str, gens: int, pop: int, seed: int) -> pd.DataFrame:
    """Extract a DataFrame [time, neg_profit, algo, instance, gens, pop, seed] from a pymoo Result."""
    F = res.pop.get("F")
    df = pd.DataFrame(F, columns=["time", "neg_profit"])
    df["algo"] = algo_name
    df["instance"] = instance_name
    df["gens"] = gens
    df["pop"] = pop
    df["seed"] = seed
    return df

def run_for_seed(inst_path: Path, gens: int, pop: int, seed: int, outdir: Path, prefix: str) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    inst_name = Path(inst_path).name

    res_nsga2 = run_nsga2(str(inst_path), n_gen=gens, pop_size=pop, seed=seed)
    res_moead = run_moead(str(inst_path), n_gen=gens, pop_size=pop, seed=seed)
    res_nsga3 = run_nsga3(str(inst_path), n_gen=gens, pop_size=pop, seed=seed)

    df2 = _df_from_res(res_nsga2, "nsga2", inst_name, gens, pop, seed)
    dfd = _df_from_res(res_moead, "moead", inst_name, gens, pop, seed)
    df3 = _df_from_res(res_nsga3, "nsga3", inst_name, gens, pop, seed)

    tag = f"{prefix}s{seed:02d}_"
    (outdir / f"{tag}front_nsga2.csv").write_text(df2.to_csv(index=False))
    (outdir / f"{tag}front_moead.csv").write_text(dfd.to_csv(index=False))
    (outdir / f"{tag}front_nsga3.csv").write_text(df3.to_csv(index=False))

    combined = pd.concat([df2, dfd, df3], ignore_index=True)
    (outdir / f"{tag}fronts_combined.csv").write_text(combined.to_csv(index=False))

    print(f"Wrote { (outdir / f'{tag}fronts_combined.csv').as_posix() }")
    return combined

def main():
    ap = argparse.ArgumentParser(description="Multi-objective TTP over multiple seeds (NSGA-II / MOEA/D / NSGA-III).")
    ap.add_argument("--inst", default="data/a280-n279.txt", help="Path to TTP instance file")
    ap.add_argument("--gens", type=int, default=250, help="Number of generations")
    ap.add_argument("--pop", type=int, default=160, help="Population size")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1], help="Random seed list: --seeds 1 2 3 4 5")
    ap.add_argument("--prefix", default="", help="Filename prefix, e.g., a280_ or fnl_")
    ap.add_argument("--outdir", default="results", help="Output directory")
    args = ap.parse_args()

    inst_path = Path(args.inst)
    outdir = Path(args.outdir)

    master = []
    for s in args.seeds:
        df_seed = run_for_seed(inst_path, gens=args.gens, pop=args.pop, seed=s, outdir=outdir, prefix=args.prefix)
        master.append(df_seed)

    master_df = pd.concat(master, ignore_index=True)
    master_path = outdir / f"{args.prefix}fronts_combined_all_seeds.csv"
    master_path.write_text(master_df.to_csv(index=False))
    print(f"Wrote {master_path.as_posix()}")

if __name__ == "__main__":
    main()
