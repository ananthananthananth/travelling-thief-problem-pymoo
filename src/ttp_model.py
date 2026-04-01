import numpy as np

def _get_profits(inst) -> np.ndarray:
    if hasattr(inst, "profits") and inst.profits is not None:
        return inst.profits.astype(np.float32, copy=False)
    return inst.items[:, 0].astype(np.float32, copy=False)

def _get_weights(inst) -> np.ndarray:
    if hasattr(inst, "weights") and inst.weights is not None:
        return inst.weights.astype(np.float32, copy=False)
    return inst.items[:, 1].astype(np.float32, copy=False)

def _get_item_city(inst) -> np.ndarray:
    if hasattr(inst, "item_city") and inst.item_city is not None:
        return inst.item_city.astype(np.int32, copy=False)
    return (inst.items[:, 2].astype(np.int32, copy=False) - 1)

def tour_time(inst, tour: np.ndarray, pick_mask: np.ndarray) -> float:
    W = _get_weights(inst)                 
    item_city = _get_item_city(inst)       
    D = inst.dist                          
    cap = float(inst.capacity)
    vmin = float(inst.v_min)
    vmax = float(inst.v_max)
    n_cities = int(inst.n_cities)

    tour = np.asarray(tour, dtype=np.int32)
    pick_mask = np.asarray(pick_mask)

    gains = np.zeros(n_cities, dtype=np.float32)
    picked_idx = np.nonzero(pick_mask > 0)[0]
    if picked_idx.size:
        np.add.at(gains, item_city[picked_idx], W[picked_idx])

    L = D[tour[:-1], tour[1:]].astype(np.float64, copy=False)

    g_tour = gains[tour[:-1]].astype(np.float64, copy=False)
    cum_w = np.cumsum(g_tour, dtype=np.float64)

    load_ratio = np.minimum(cum_w / max(cap, 1e-12), 1.0)
    speeds = vmax - (vmax - vmin) * load_ratio
    times = L / np.maximum(speeds, 1e-12)

    return float(times.sum())

def profit(inst, pick_mask: np.ndarray) -> float:
    P = _get_profits(inst)
    pick_mask = np.asarray(pick_mask, dtype=P.dtype)
    return float((P * pick_mask).sum())

def objectives(inst, tour: np.ndarray, pick_mask: np.ndarray):

    t = tour_time(inst, tour, pick_mask)
    p = profit(inst, pick_mask)
    net = p - float(inst.rent) * t
    return t, -p, net
