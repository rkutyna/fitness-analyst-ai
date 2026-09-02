"""Nutrition reference catalog: item -> macros per serving, with provenance.

A lookup table so that estimating a meal is arithmetic over recorded numbers
instead of the model recalling nutrition facts. Two rules carry the design:

1. **Every row says where its numbers came from.** `source` is one of four
   tiers and `source_detail` is required, the way `manual_jog.why` is required.
   A catalog the model populates is in tension with "Python owns the truth";
   the tier is how that tension is resolved. The model may propose a row, it
   may never propose one silently.

2. **`add` is a full replace.** `subjective.log` is a partial upsert because a
   check-in arrives as drip-fed corrections. A catalog row is the opposite: one
   complete statement read off one label. Under partial semantics a row could
   be relabelled `estimate` -> `web` while its omitted macros kept their
   guessed values, and the row would then read as web-sourced throughout.

`totals` exists so the model never multiplies servings by calories itself —
that is derivation, which this project reserves for Python.
"""
from __future__ import annotations

import math
import sqlite3

from . import db

# Canonical tier ordering, WORST FIRST. This tuple is the single definition of
# rank: `weakest_source` is the minimum index into it. Stating the order twice
# in opposite directions is how you end up implementing "weakest" backwards.
SOURCES_WORST_TO_BEST = ("estimate", "web", "label_text", "label_photo")

MACROS = ("protein_g", "carb_g", "fat_g")

# SQLite LIKE treats % and _ as wildcards, so an unescaped query of "%" matches
# the whole catalog and "100% whey" matches everything containing "100".
_LIKE_ESCAPE = "\\"


def _finite(name: str, value) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return v


def _non_negative(name: str, value):
    if value is None:
        return None
    v = _finite(name, value)
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def _text(name: str, value, *, required: bool) -> str | None:
    s = (value or "").strip()
    if not s:
        if required:
            raise ValueError(f"{name} is required")
        return None
    return s


def _validate(fields: dict) -> dict:
    key = (fields.get("item_key") or "").strip()
    if not key or key != key.lower() or any(c.isspace() for c in key):
        raise ValueError("item_key must be a non-empty lowercase slug with no whitespace")
    fields["item_key"] = key

    for name in ("display_name", "serving_desc", "source_detail"):
        fields[name] = _text(name, fields.get(name), required=True)
    for name in ("brand", "aliases", "notes"):
        fields[name] = _text(name, fields.get(name), required=False)

    if fields.get("source") not in SOURCES_WORST_TO_BEST:
        raise ValueError(
            f"source must be one of {SOURCES_WORST_TO_BEST}, got {fields.get('source')!r}")

    if fields.get("kcal") is None:
        raise ValueError("kcal is required")
    fields["kcal"] = _non_negative("kcal", fields["kcal"])
    for name in MACROS:
        fields[name] = _non_negative(name, fields.get(name))

    if fields.get("serving_g") is not None:
        g = _finite("serving_g", fields["serving_g"])
        if g <= 0:
            raise ValueError("serving_g must be > 0")
        fields["serving_g"] = g

    if fields.get("confirmed") not in (0, 1):
        raise ValueError("confirmed must be 0 or 1")
    return fields


def _row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r is not None else None


def get(conn: sqlite3.Connection, item_key: str) -> dict | None:
    return _row(conn.execute(
        "SELECT * FROM food_catalog WHERE item_key = ?", (item_key,)).fetchone())


def search(conn: sqlite3.Connection, query: str = "") -> list[dict]:
    """Case-insensitive CONTAINS-search over display_name and aliases.

    An empty or whitespace-only query returns the whole catalog — that is the
    audit path. Case-insensitivity is ASCII-only: SQLite's default LIKE does
    not case-fold non-ASCII, so 'cafe' and 'CAFE' match but 'café' and 'CAFÉ'
    do not. Accepted rather than fixed; the alternative is an ICU build.
    """
    q = (query or "").strip()
    if not q:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM food_catalog ORDER BY display_name")]
    escaped = (q.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
                .replace("%", _LIKE_ESCAPE + "%")
                .replace("_", _LIKE_ESCAPE + "_"))
    pattern = f"%{escaped}%"
    return [dict(r) for r in conn.execute(
        """SELECT * FROM food_catalog
           WHERE display_name LIKE ? ESCAPE ?
              OR COALESCE(aliases, '') LIKE ? ESCAPE ?
           ORDER BY display_name""",
        (pattern, _LIKE_ESCAPE, pattern, _LIKE_ESCAPE))]


def add(conn: sqlite3.Connection, item_key: str, display_name: str, *,
        serving_desc: str, kcal: float, source: str, source_detail: str,
        brand: str | None = None, aliases: str | None = None,
        serving_g: float | None = None, protein_g: float | None = None,
        carb_g: float | None = None, fat_g: float | None = None,
        confirmed: int = 0, notes: str | None = None) -> dict:
    """Write one catalog row, REPLACING any existing row for `item_key`.

    Every column is written from the arguments; an omitted optional field
    becomes NULL rather than keeping its previous value. See the module
    docstring for why this is not a partial upsert. Raises ValueError on
    invalid input. Commits. Returns the stored row.
    """
    fields = _validate({
        "item_key": item_key, "display_name": display_name, "brand": brand,
        "aliases": aliases, "serving_desc": serving_desc, "serving_g": serving_g,
        "kcal": kcal, "protein_g": protein_g, "carb_g": carb_g, "fat_g": fat_g,
        "source": source, "source_detail": source_detail,
        "confirmed": confirmed, "notes": notes,
    })
    fields["verified_at"] = db.utcnow_iso()
    conn.execute(
        """
        INSERT OR REPLACE INTO food_catalog
            (item_key, display_name, brand, aliases, serving_desc, serving_g,
             kcal, protein_g, carb_g, fat_g, source, source_detail, confirmed,
             verified_at, notes)
        VALUES
            (:item_key, :display_name, :brand, :aliases, :serving_desc, :serving_g,
             :kcal, :protein_g, :carb_g, :fat_g, :source, :source_detail, :confirmed,
             :verified_at, :notes)
        """,
        fields,
    )
    conn.commit()
    return get(conn, fields["item_key"])


def totals(conn: sqlite3.Connection, items: list[tuple[str, float]]) -> dict:
    """Sum a meal from `[(item_key, servings), ...]`.

    `kcal` is NOT NULL in the schema, so it always sums. The macros are
    nullable, and a partial macro sum that *looks* complete is the failure mode
    worth preventing: one item with 15 g protein plus one with unknown protein
    must not report "15 g". So per macro — if any item with servings > 0 is
    missing it, that total is None and the macro is named in `incomplete`.

    `weakest_source` is the worst tier among items with servings > 0. It is the
    honest headline for any total: a meal is only as known as the least-known
    thing in it.
    """
    if not items:
        raise ValueError("items is empty — nothing to total")

    resolved = []
    for item_key, servings in items:
        n = _finite("servings", servings)
        if n < 0:
            raise ValueError("servings must be >= 0")
        row = get(conn, item_key)
        if row is None:
            # Silently skipping would return a total that is quietly missing a
            # food. A total nobody can trust is worse than no total.
            raise ValueError(f"unknown item_key {item_key!r} — add it to the catalog first")
        resolved.append((row, n))

    contributing = [(row, n) for row, n in resolved if n > 0]
    out: dict = {"kcal": round(sum(row["kcal"] * n for row, n in contributing), 1)}
    incomplete: list[str] = []
    for macro in MACROS:
        if any(row[macro] is None for row, _ in contributing):
            out[macro] = None
            incomplete.append(macro)
        else:
            out[macro] = round(sum(row[macro] * n for row, n in contributing), 1)
    out["incomplete"] = incomplete
    out["weakest_source"] = min(
        (row["source"] for row, _ in contributing),
        key=SOURCES_WORST_TO_BEST.index, default=None)
    out["items"] = [
        {"item_key": row["item_key"], "display_name": row["display_name"],
         "servings": n, "serving_desc": row["serving_desc"],
         "kcal": round(row["kcal"] * n, 1), "source": row["source"],
         "confirmed": row["confirmed"]}
        for row, n in resolved
    ]
    return out
