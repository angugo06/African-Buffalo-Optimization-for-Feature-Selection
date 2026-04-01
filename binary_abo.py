import numpy as np
import random
from typing import Optional, Dict, Any, Callable

def ensure_nonempty(mask: np.ndarray, min_k: int = 1, rng: random.Random = None):
    if rng is None:
        rng = random.Random()
    k = int(mask.sum())
    p = len(mask)
    if k >= min_k:
        return mask
    idxs = list(range(p))
    rng.shuffle(idxs)
    for j in idxs:
        if mask[j] == 0:
            mask[j] = 1
            k += 1
            if k >= min_k:
                break
    return mask

def random_mask(p: int, density: float, rng: random.Random) -> np.ndarray:
    mask = np.zeros(p, dtype=np.uint8)
    for j in range(p):
        if rng.random() < density:
            mask[j] = 1
    return ensure_nonempty(mask, min_k=1, rng=rng)

def flip_bits(mask: np.ndarray, flip_rate: float, rng: random.Random) -> np.ndarray:
    new_mask = mask.copy()
    for j in range(len(mask)):
        if rng.random() < flip_rate:
            new_mask[j] = 1 - new_mask[j]
    return ensure_nonempty(new_mask, min_k=1, rng=rng)

def guided_move(mask: np.ndarray, pbest_mask: np.ndarray, gbest_mask: np.ndarray,
                guided_frac: float, rng: random.Random) -> np.ndarray:
    p = len(mask)
    new_mask = mask.copy()
    m = max(1, int(round(guided_frac * p)))
    idxs = list(range(p))
    rng.shuffle(idxs)
    for j in idxs[:m]:
        desire_one = (pbest_mask[j] == 1) or (gbest_mask[j] == 1)
        if desire_one:
            if rng.random() < 0.8: new_mask[j] = 1
        else:
            if rng.random() < 0.8: new_mask[j] = 0
    return ensure_nonempty(new_mask, min_k=1, rng=rng)

class BinaryABO:
    def __init__(self, p: int, score_fn: Callable[[np.ndarray], float],
                 herd_size: int = 60, iterations: int = 200,
                 p_explore: float = 0.6, flip_rate: float = 0.05,
                 guided_frac: float = 0.2, init_density: float = 0.3,
                 seed: Optional[int] = None):
        self.p = p
        self.score_fn = score_fn
        self.herd_size = herd_size
        self.iterations = iterations
        self.p_explore = p_explore
        self.flip_rate = flip_rate
        self.guided_frac = guided_frac
        self.init_density = init_density
        self.rng = random.Random(seed)

    def solve(self) -> Dict[str, Any]:
        herd = [random_mask(self.p, self.init_density, self.rng) for _ in range(self.herd_size)]
        pbest = [m.copy() for m in herd]
        pbest_score = [self.score_fn(m) for m in pbest]
        g_idx = int(np.argmax(pbest_score))
        gbest = pbest[g_idx].copy()
        gbest_score = pbest_score[g_idx]
        history = [gbest_score]

        for _ in range(self.iterations):
            new_herd = []
            for i, m in enumerate(herd):
                if self.rng.random() < self.p_explore:
                    cand = flip_bits(m, self.flip_rate, self.rng)       # exploración
                else:
                    cand = guided_move(m, pbest[i], gbest, self.guided_frac, self.rng)  # explotación
                if self.score_fn(cand) > self.score_fn(m):
                    new_herd.append(cand)
                else:
                    new_herd.append(m)
            herd = new_herd

            for i, m in enumerate(herd):
                s = self.score_fn(m)
                if s > pbest_score[i]:
                    pbest[i] = m.copy()
                    pbest_score[i] = s
                    if s > gbest_score:
                        gbest, gbest_score = m.copy(), s
            history.append(gbest_score)

        return {"gbest_mask": gbest, "gbest_score": gbest_score, "history": history}
