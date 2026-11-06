import numpy as np

def _minmax(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x), dtype=float)
    v = x[ok]
    if v.size == 0:
        return out
    spread = np.ptp(v)
    # constant fallback keeps the 0-1 scale (was: raw values)
    out[ok] = 1.0 if (v.size < 2 or spread < 1e-9) else (v - v.min()) / spread
    return out

def fuse(s, tag_weight: float, top_n: int) -> list[dict]:
    if not 0.0 <= tag_weight <= 1.0:
        raise ValueError("tag_weight must be in [0, 1]")

    tn, pn = _minmax(s.tag, s.tag_ok), _minmax(s.plot, s.plot_ok)

    final = tag_weight * tn + (1 - tag_weight) * pn   # missing signal contributes 0

    final[~(s.tag_ok | s.plot_ok)] = -1.0             # neither signal: drop
    order = np.argsort(-final, kind="stable")[:top_n]
    return [
        {
            "movie_id": s.ids[i],
            "final_score": float(final[i]),
            "tag_score": float(tn[i]) if s.tag_ok[i] else None,
            "plot_score": float(pn[i]) if s.plot_ok[i] else None,
        }
        for i in order if final[i] >= 0
    ]