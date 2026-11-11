import numpy as np

def _scale(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    # raw cosine clipped to 0-1; missing/empty vector -> 0
    return np.where(ok, np.clip(x, 0.0, 1.0), 0.0)

def fuse(s, tag_weight: float, top_n: int) -> list[dict]:
    if not 0.0 <= tag_weight <= 1.0:
        raise ValueError("tag_weight must be in [0, 1]")

    tn, pn = _scale(s.tag, s.tag_ok), _scale(s.plot, s.plot_ok)

    final = tag_weight * tn + (1 - tag_weight) * pn

    final[~(s.tag_ok | s.plot_ok)] = -1.0             # neither signal: drop
    order = np.lexsort((np.asarray(s.ids), -final))[:top_n]
    return [
        {
            "movie_id": s.ids[i],
            "final_score": float(final[i]),
            "tag_score": float(tn[i]),
            "plot_score": float(pn[i]),
        }
        for i in order if final[i] >= 0
    ]