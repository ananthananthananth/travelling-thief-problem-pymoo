from pathlib import Path
import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# SOO (GA)
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

# SOO (DE)
from pymoo.algorithms.soo.nonconvex.de import DE

# MOO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions

from .ttp_io import load_ttp_instance
from .ttp_model import objectives

def _weights(inst) -> np.ndarray:
    if hasattr(inst, "weights") and inst.weights is not None:
        return np.asarray(inst.weights, dtype=np.float32)
    return np.asarray(inst.items[:, 1], dtype=np.float32)

def _profits(inst) -> np.ndarray:
    if hasattr(inst, "profits") and inst.profits is not None:
        return np.asarray(inst.profits, dtype=np.float32)
    return np.asarray(inst.items[:, 0], dtype=np.float32)

def _decode(inst, x):
    n, m = inst.n_cities, inst.n_items
    x = np.asarray(x)

    tour = np.argsort(x[:n])                         
    pick = (x[n:] > 0.5).astype(np.int32)           

    W = _weights(inst)
    P = _profits(inst)

    w = float((W * pick).sum())
    if w > inst.capacity:
        ratio = P / np.maximum(W, 1e-9)
        chosen = np.where(pick == 1)[0]
        to_consider = chosen[np.argsort(ratio[chosen])]
        for k in to_consider:
            pick[k] = 0
            w -= float(W[k])
            if w <= inst.capacity:
                break

    return tour.astype(np.int32), pick.astype(np.float32)

class TTP_MO(Problem):
    def __init__(self, inst_path: str):
        self.inst = load_ttp_instance(inst_path)
        n_var = self.inst.n_cities + self.inst.n_items
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        N = X.shape[0]
        F = np.empty((N, 2), dtype=float)

        for i in range(N):
            tour, pick = _decode(self.inst, X[i])
            t, neg_p, _ = objectives(self.inst, tour, pick)
            F[i, 0] = t
            F[i, 1] = neg_p
        out["F"] = F

class TTP_SO(Problem):
    def __init__(self, inst_path: str, w_time: float, w_profit: float,
                 norm_samples: int = 50, seed: int = 1):
        assert abs((w_time + w_profit) - 1.0) < 1e-9, "weights must sum to 1"
        self.inst = load_ttp_instance(inst_path)
        self.w_time = float(w_time)
        self.w_profit = float(w_profit)

        rng = np.random.RandomState(seed)
        n, m = self.inst.n_cities, self.inst.n_items
        t_vals, p_vals = [], []
        for _ in range(norm_samples):
            x = rng.rand(n + m)
            tour, pick = _decode(self.inst, x)
            t, neg_p, _ = objectives(self.inst, tour, pick)
            t_vals.append(t)
            p_vals.append(-neg_p)
        self.t_min, self.t_max = float(np.min(t_vals)), float(np.max(t_vals))
        self.p_min, self.p_max = float(np.min(p_vals)), float(np.max(p_vals))

        super().__init__(n_var=n + m, n_obj=1, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        N = X.shape[0]
        F = np.empty((N, 1), dtype=float)

        dt = max(self.t_max - self.t_min, 1e-12)
        dp = max(self.p_max - self.p_min, 1e-12)

        for i in range(N):
            tour, pick = _decode(self.inst, X[i])
            t, neg_p, _ = objectives(self.inst, tour, pick)
            p = -neg_p

            t_n = (t - self.t_min) / dt
            p_loss = (self.p_max - p) / dp
            F[i, 0] = self.w_time * t_n + self.w_profit * p_loss

        out["F"] = F

def run_ga_single(inst_path, w_time, w_profit, n_gen=200, pop_size=120, seed=1,
                  eta_c=15, eta_m=20):
    problem = TTP_SO(inst_path, w_time, w_profit, norm_samples=50, seed=seed)

    algo = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=eta_c, prob=1.0),
        mutation=PolynomialMutation(eta=eta_m, prob=1.0 / problem.n_var),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algo,
                   termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=True)

    tour, pick = _decode(problem.inst, res.X)
    t, neg_p, net = objectives(problem.inst, tour, pick)
    return {
        "best_f": float(res.F),
        "time": float(t),
        "profit": float(-neg_p),
        "net": float(net),
        "X": res.X,
    }

def run_de_single(inst_path, w_time, w_profit, n_gen=200, pop_size=120, seed=1,
                  F=0.5, CR=0.9, variant="DE/rand/1/bin"):
    problem = TTP_SO(inst_path, w_time, w_profit, norm_samples=50, seed=seed)

    algo = DE(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        variant=variant, 
        F=F,
        CR=CR,
    )

    res = minimize(problem, algo,
                   termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=True)

    tour, pick = _decode(problem.inst, res.X)
    t, neg_p, net = objectives(problem.inst, tour, pick)
    return {
        "best_f": float(res.F),
        "time": float(t),
        "profit": float(-neg_p),
        "net": float(net),
        "X": res.X,
    }

def run_nsga2(inst_path, n_gen=250, pop_size=160, seed=1):
    problem = TTP_MO(inst_path)
    algo = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=15),
        mutation=PolynomialMutation(prob=1.0 / problem.n_var, eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(problem, algo, get_termination("n_gen", n_gen), seed=seed, verbose=True)
    return res

def run_moead(inst_path, n_gen=250, pop_size=160, seed=1):
    problem = TTP_MO(inst_path)
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=pop_size, seed=seed)
    algo = MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    res = minimize(problem, algo, get_termination("n_gen", n_gen), seed=seed, verbose=True)
    return res

def run_nsga3(inst_path, n_gen=250, pop_size=160, seed=1):
    problem = TTP_MO(inst_path)
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=pop_size, seed=seed)
    algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    res = minimize(problem, algo, get_termination("n_gen", n_gen), seed=seed, verbose=True)
    return res
