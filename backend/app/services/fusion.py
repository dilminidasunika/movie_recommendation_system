import numpy as np
from app.config import settings


def _scale(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    # raw cosine clipped to 0-1; missing/empty vector -> 0
    return np.where(ok, np.clip(x, 0.0, 1.0), 0.0)


def fuse(s, tag_weight: float, top_n: int, tie_break_plot: bool | None = None) -> list[dict]:
    if not 0.0 <= tag_weight <= 1.0:
        raise ValueError("tag_weight must be in [0, 1]")

    tn, pn = _scale(s.tag, s.tag_ok), _scale(s.plot, s.plot_ok)

    final = tag_weight * tn + (1 - tag_weight) * pn

    final[~(s.tag_ok | s.plot_ok)] = -1.0             # neither signal: drop

    if tie_break_plot is None:
        tie_break_plot = getattr(settings, "tie_break_plot", False)

    ids = np.asarray(s.ids)
    if tie_break_plot:
        # order by final score desc, plot score desc, id asc
        order = np.lexsort((ids, -pn, -final))[:top_n]
    else:
        order = np.lexsort((ids, -final))[:top_n]

    return [
        {
            "movie_id": int(s.ids[i]),
            "final_score": float(final[i]),
            "tag_score": float(tn[i]),
            "plot_score": float(pn[i]),
        }
        for i in order if final[i] >= 0
    ]
