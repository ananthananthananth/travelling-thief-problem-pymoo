from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from pathlib import Path

@dataclass
class TTPInstance:
    name: str
    n_cities: int
    n_items: int
    capacity: float
    v_min: float
    v_max: float
    rent: float
    coords: np.ndarray          
    items: np.ndarray           
    dist: np.ndarray            

def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def compute_dist_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=np.float32)   #float32
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclid(coords[i], coords[j])
            D[i, j] = D[j, i] = d
    return D

def load_ttp_instance(path: str) -> TTPInstance:
    p = Path(path)
    text = p.read_text().strip().splitlines()
    name = p.stem

    n_cities = int([line for line in text if line.startswith("DIMENSION")][0].split(":")[1])
    n_items = int([line for line in text if line.startswith("NUMBER OF ITEMS")][0].split(":")[1])
    capacity = float([line for line in text if line.startswith("CAPACITY OF KNAPSACK")][0].split(":")[1])
    v_min = float([line for line in text if line.startswith("MIN SPEED")][0].split(":")[1])
    v_max = float([line for line in text if line.startswith("MAX SPEED")][0].split(":")[1])
    rent = float([line for line in text if line.startswith("RENTING RATIO")][0].split(":")[1])

    coords_start = text.index(next(l for l in text if l.startswith("NODE_COORD_SECTION")))
    items_start = text.index(next(l for l in text if l.startswith("ITEMS SECTION")))

    coords_lines = text[coords_start + 1: items_start]
    coords = np.array([[float(x), float(y)] for _, x, y in (line.split() for line in coords_lines)], dtype=np.float32)

    items_lines = text[items_start + 1:]
    items = np.array([[float(pf), float(wt), int(city)] for _, pf, wt, city in (line.split() for line in items_lines)], dtype=np.float32)

    dist = compute_dist_matrix(coords)

    return TTPInstance(
        name=name,
        n_cities=n_cities,
        n_items=n_items,
        capacity=capacity,
        v_min=v_min,
        v_max=v_max,
        rent=rent,
        coords=coords,
        items=items,
        dist=dist,
    )
