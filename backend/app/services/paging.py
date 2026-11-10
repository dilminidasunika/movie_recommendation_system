import hashlib
import json
import threading

from cachetools import TTLCache

_cache: TTLCache = TTLCache(maxsize=200, ttl=900)   # 15 min, RAM only, no user identity
_lock = threading.Lock()


def request_id(req) -> str:
    d = req.model_dump(mode="json", exclude={"top_n"})   # page size is not part of identity
    d["movie_ids"] = sorted(d.get("movie_ids") or [])
    if "tags" in d and isinstance(d["tags"], list):
        d["tags"] = sorted(d["tags"], key=lambda t: (t.get("tag_type", ""), t.get("name", "")))
    if "genres" in d and isinstance(d["genres"], list):
        d["genres"] = sorted(d["genres"])
    if "original_language" in d and isinstance(d["original_language"], list):
        d["original_language"] = sorted(d["original_language"])
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def get(rid: str):
    with _lock:
        entry = _cache.get(rid)
        if entry is not None:
            _cache[rid] = entry      # re-set resets the TTL clock (sliding expiry)
        return entry


def put(rid: str, entry: dict) -> None:
    with _lock:
        _cache[rid] = entry
