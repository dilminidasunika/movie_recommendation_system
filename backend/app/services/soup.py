# app/services/soup.py
"""Single source of truth for tag soup. Imported by scripts/ (offline) AND query_builder (runtime)."""
import re

# Order = column-block order of the saved matrix. MUST match your fitted vectorizers.
HIGH_FIELDS = ['cast_members', 'genres', 'spoken_languages', 'production_countries']
MID_FIELDS  = ['producers', 'writers', 'music_composer']
LOW_FIELDS  = ['director', 'keywords']

TIERS: list[list[str]] = [
    HIGH_FIELDS,   # tier 0 — tag_vectorizer_high
    MID_FIELDS,    # tier 1 — tag_vectorizer_mid
    LOW_FIELDS,    # tier 2 — tag_vectorizer_low
]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def squash(name: str) -> str:
    """'Clint Eastwood' -> 'clinteastwood'. Must be identical offline and online."""
    return _NON_ALNUM.sub("", str(name).lower())


def build_tier_docs(fields: dict[str, list[str]]) -> list[str]:
    """fields: {field_name: [values]} (a full movie row or a partial one from chips).
    Returns one space-separated doc per tier."""
    docs = []
    for tier_fields in TIERS:
        toks = []
        for f in tier_fields:
            toks.extend(t for t in (squash(v) for v in fields.get(f, [])) if t)
        docs.append(" ".join(toks))
    return docs