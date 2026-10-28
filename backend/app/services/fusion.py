# services/fusion.py
import numpy as np

EPS = 1e-6

def _minmax(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x), dtype=float)
    v = x[ok]
    if v.size == 0:
        return out
    spread = np.ptp(v)
    out[ok] = v if (v.size < 2 or spread < 1e-9) else (v - v.min()) / spread  # raw fallback
    return out

def fuse(s, tag_weight: float, top_n: int) -> list[dict]:
    tn, pn = _minmax(s.tag, s.tag_ok), _minmax(s.plot, s.plot_ok)

    # EPS keeps denom > 0 when tag_weight is exactly 0 or 1 and a
    # candidate only has the "zero-weighted" signal.
    wt = np.where(s.tag_ok, tag_weight + EPS, 0.0)
    wp = np.where(s.plot_ok, (1 - tag_weight) + EPS, 0.0)
    denom = wt + wp
    final = np.divide(wt * tn + wp * pn, denom, out=np.zeros_like(denom), where=denom > 0)

    # candidates with neither signal are dropped
    final[denom == 0] = -1.0
    order = np.argsort(-final)[:top_n]
    return [
        {
            "movie_id": s.ids[i],
            "final_score": float(final[i]),
            "tag_score": float(tn[i]) if s.tag_ok[i] else None,
            "plot_score": float(pn[i]) if s.plot_ok[i] else None,
        }
        for i in order if final[i] >= 0
    ]