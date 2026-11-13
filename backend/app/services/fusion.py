import numpy as np
from app.config import settings


def _scale(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    # raw cosine clipped to 0-1; missing/empty vector -> 0
    return np.where(ok, np.clip(x, 0.0, 1.0), 0.0)


def compute_final_scores(s, tag_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute final scores in float64 precision. Returns (final, tn, pn)."""
    tn = _scale(s.tag, s.tag_ok).astype(np.float64)
    pn = _scale(s.plot, s.plot_ok).astype(np.float64)
    tw = float(tag_weight)
    final = tw * tn + (1.0 - tw) * pn
    final[~(s.tag_ok | s.plot_ok)] = -1.0
    return final, tn, pn


def fuse(s, tag_weight: float, top_n: int, tie_break_plot: bool | None = None) -> list[dict]:
    if not 0.0 <= tag_weight <= 1.0:
        raise ValueError("tag_weight must be in [0, 1]")

    final, tn, pn = compute_final_scores(s, tag_weight)

    if tie_break_plot is None:
        tie_break_plot = getattr(settings, "tie_break_plot", False)

    valid = np.flatnonzero(final > 0)
    if len(valid) == 0:
        return []

    ids = np.asarray(s.ids)
    sub_ids = ids[valid]
    sub_final = final[valid]
    sub_pn = pn[valid]
    sub_tn = tn[valid]

    if tie_break_plot:
        # order by final score desc, plot score desc, id asc
        sub_order = np.lexsort((sub_ids, -sub_pn, -sub_final))[:top_n]
    else:
        sub_order = np.lexsort((sub_ids, -sub_final))[:top_n]

    chosen = valid[sub_order]

    return [
        {
            "movie_id": int(ids[i]),
            "final_score": round(float(final[i]), 6),
            "tag_score": round(float(tn[i]), 6),
            "plot_score": round(float(pn[i]), 6),
        }
        for i in chosen
    ]
