"""
M3 - Pair Feature Engineering

Converts M2 candidate pairs into pair-level matching features.

Input:
    - Source 1 dataframe
    - Source 2/3 candidate dataframes
    - M2 candidate dataframe

Output:
    One row per candidate pair with similarity, agreement,
    and contradiction features.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _safe_text(value) -> str:
    """Convert a value to a normalized string representation."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _tokens(text: str) -> set[str]:
    """Return alphanumeric tokens from a text value."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _numeric_tokens(text: str) -> set[str]:
    """Extract numeric tokens from an address/name."""
    return set(re.findall(r"\d+", text.lower()))


def _jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity."""
    a = _tokens(text_a)
    b = _tokens(text_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _containment(text_a: str, text_b: str) -> float:
    """
    Calculate directional token containment.

    Example:
        A = "abc motors pune"
        B = "abc motors"
        containment = 1.0
    """
    a = _tokens(text_a)
    b = _tokens(text_b)

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def _sequence_similarity(text_a: str, text_b: str) -> float:
    """Character-level similarity using SequenceMatcher."""
    a = _safe_text(text_a).lower()
    b = _safe_text(text_b).lower()

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def _length_difference(text_a: str, text_b: str) -> int:
    """Absolute character-length difference."""
    return abs(len(_safe_text(text_a)) - len(_safe_text(text_b)))


def _token_count_difference(text_a: str, text_b: str) -> int:
    """Absolute token-count difference."""
    return abs(len(_tokens(text_a)) - len(_tokens(text_b)))


def _exact_match(text_a: str, text_b: str) -> int:
    """Exact non-empty string match."""
    a = _safe_text(text_a).lower()
    b = _safe_text(text_b).lower()

    return int(bool(a) and bool(b) and a == b)


def _numeric_overlap(text_a: str, text_b: str) -> float:
    """Numeric-token overlap between two strings."""
    a = _numeric_tokens(text_a)
    b = _numeric_tokens(text_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _numeric_conflict(text_a: str, text_b: str) -> int:
    """
    Detect conflicting numeric address information.

    A conflict is raised when both records contain numbers but
    none of the numbers overlap.
    """
    a = _numeric_tokens(text_a)
    b = _numeric_tokens(text_b)

    return int(bool(a) and bool(b) and not (a & b))


# ---------------------------------------------------------------------
# Individual field feature builders
# ---------------------------------------------------------------------

def _name_features(s1: pd.Series, candidate: pd.Series) -> dict:
    """Build name-related features."""

    name = _safe_text(s1.get("name_normalized", ""))
    cand_name = _safe_text(candidate.get("name_normalized", ""))

    compact = _safe_text(s1.get("name_compact", ""))
    cand_compact = _safe_text(candidate.get("name_compact", ""))

    no_suffix = _safe_text(s1.get("name_no_suffix", ""))
    cand_no_suffix = _safe_text(candidate.get("name_no_suffix", ""))

    no_suffix_compact = _safe_text(
        s1.get("name_no_suffix_compact", "")
    )
    cand_no_suffix_compact = _safe_text(
        candidate.get("name_no_suffix_compact", "")
    )

    dedup = _safe_text(s1.get("name_dedup", ""))
    cand_dedup = _safe_text(candidate.get("name_dedup", ""))

    prefix = _safe_text(s1.get("name_prefix", ""))
    cand_prefix = _safe_text(candidate.get("name_prefix", ""))

    prefix_6 = _safe_text(s1.get("name_prefix_6", ""))
    cand_prefix_6 = _safe_text(candidate.get("name_prefix_6", ""))

    first_token = _safe_text(s1.get("name_first_token", ""))
    cand_first_token = _safe_text(candidate.get("name_first_token", ""))

    return {
        "name_exact": _exact_match(name, cand_name),
        "name_compact_exact": _exact_match(compact, cand_compact),
        "name_no_suffix_exact": _exact_match(
            no_suffix, cand_no_suffix
        ),
        "name_no_suffix_compact_exact": _exact_match(
            no_suffix_compact,
            cand_no_suffix_compact,
        ),
        "name_dedup_exact": _exact_match(dedup, cand_dedup),

        "name_jaccard": _jaccard(name, cand_name),
        "name_compact_jaccard": _jaccard(
            compact,
            cand_compact,
        ),
        "name_no_suffix_jaccard": _jaccard(
            no_suffix,
            cand_no_suffix,
        ),

        "name_containment": _containment(name, cand_name),
        "name_char_similarity": _sequence_similarity(
            name,
            cand_name,
        ),

        "name_length_diff": _length_difference(
            name,
            cand_name,
        ),
        "name_token_count_diff": _token_count_difference(
            name,
            cand_name,
        ),

        "name_prefix_match": _exact_match(prefix, cand_prefix),
        "name_prefix_6_match": _exact_match(
            prefix_6,
            cand_prefix_6,
        ),
        "name_first_token_match": _exact_match(
            first_token,
            cand_first_token,
        ),
    }


def _address_features(s1: pd.Series, candidate: pd.Series) -> dict:
    """Build address-related features."""

    address = _safe_text(s1.get("address_normalized", ""))
    cand_address = _safe_text(
        candidate.get("address_normalized", "")
    )

    compact = _safe_text(s1.get("address_compact", ""))
    cand_compact = _safe_text(
        candidate.get("address_compact", "")
    )

    return {
        "address_exact": _exact_match(
            address,
            cand_address,
        ),
        "address_compact_exact": _exact_match(
            compact,
            cand_compact,
        ),

        "address_jaccard": _jaccard(
            address,
            cand_address,
        ),
        "address_containment": _containment(
            address,
            cand_address,
        ),
        "address_char_similarity": _sequence_similarity(
            address,
            cand_address,
        ),

        "address_length_diff": _length_difference(
            address,
            cand_address,
        ),
        "address_token_count_diff": _token_count_difference(
            address,
            cand_address,
        ),

        "address_numeric_overlap": _numeric_overlap(
            address,
            cand_address,
        ),
        "address_numeric_conflict": _numeric_conflict(
            address,
            cand_address,
        ),

        "address_first_token_match": _exact_match(
            _safe_text(s1.get("address_first_token", "")),
            _safe_text(candidate.get("address_first_token", "")),
        ),
    }


def _country_features(s1: pd.Series, candidate: pd.Series) -> dict:
    """Build country-related features."""

    country = _safe_text(
        s1.get("country_normalized", "")
    ).lower()

    cand_country = _safe_text(
        candidate.get("country_normalized", "")
    ).lower()

    has_both = bool(country) and bool(cand_country)

    return {
        "country_match": int(
            has_both and country == cand_country
        ),
        "country_conflict": int(
            has_both and country != cand_country
        ),
    }


# ---------------------------------------------------------------------
# Evidence / contradiction features
# ---------------------------------------------------------------------

def _evidence_features(
    name_features: dict,
    address_features: dict,
    country_features: dict,
) -> dict:
    """Aggregate supporting and contradictory evidence."""

    name_support = sum(
        [
            name_features["name_exact"],
            name_features["name_compact_exact"],
            name_features["name_no_suffix_exact"],
            name_features["name_no_suffix_compact_exact"],
            name_features["name_dedup_exact"],
            name_features["name_prefix_match"],
            name_features["name_prefix_6_match"],
            name_features["name_first_token_match"],
        ]
    )

    address_support = sum(
        [
            address_features["address_exact"],
            address_features["address_compact_exact"],
            address_features["address_first_token_match"],
        ]
    )

    contradiction_count = sum(
        [
            address_features["address_numeric_conflict"],
            country_features["country_conflict"],
        ]
    )

    return {
        "name_support_count": name_support,
        "address_support_count": address_support,
        "total_support_count": name_support + address_support,

        "contradiction_count": contradiction_count,

        "strong_name_weak_address": int(
            name_features["name_char_similarity"] >= 0.85
            and address_features["address_jaccard"] < 0.20
        ),

        "strong_name_numeric_conflict": int(
            name_features["name_char_similarity"] >= 0.85
            and address_features["address_numeric_conflict"] == 1
        ),

        "country_conflict_with_name_match": int(
            country_features["country_conflict"] == 1
            and name_features["name_char_similarity"] >= 0.85
        ),
    }


# ---------------------------------------------------------------------
# Main pair feature builder
# ---------------------------------------------------------------------

def build_pair_features(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build M3 features for M2 candidate pairs.

    Parameters
    ----------
    source1:
        Preprocessed Source 1 dataframe.

    candidate_sources:
        Dictionary such as:
            {
                "source2": source2_dataframe,
                "source3": source3_dataframe,
            }

    candidates:
        M2 candidate dataframe containing:
            source1_id
            candidate_id
            candidate_source

    Returns
    -------
    pd.DataFrame
        One row per candidate pair.
    """

    required_candidate_columns = {
        "source1_id",
        "candidate_id",
        "candidate_source",
    }

    missing = required_candidate_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Candidate dataframe is missing columns: {sorted(missing)}"
        )

    if "entity_id" not in source1.columns:
        raise ValueError(
            "Source 1 dataframe must contain 'entity_id'."
        )

    source1_lookup = source1.copy()
    source1_lookup["entity_id"] = (
        source1_lookup["entity_id"].astype(str)
    )

    source1_lookup = source1_lookup.set_index("entity_id")

    candidate_lookups: dict[str, pd.DataFrame] = {}

    for source_name, dataframe in candidate_sources.items():

        if "entity_id" not in dataframe.columns:
            raise ValueError(
                f"{source_name} dataframe must contain 'entity_id'."
            )

        lookup = dataframe.copy()

        lookup["entity_id"] = (
            lookup["entity_id"].astype(str)
        )

        lookup = lookup.set_index("entity_id")

        candidate_lookups[source_name] = lookup

    rows: list[dict] = []

    for _, pair in candidates.iterrows():

        source1_id = str(pair["source1_id"])
        candidate_id = str(pair["candidate_id"])
        candidate_source = str(pair["candidate_source"])

        if source1_id not in source1_lookup.index:
            continue

        if candidate_source not in candidate_lookups:
            continue

        candidate_lookup = candidate_lookups[candidate_source]

        if candidate_id not in candidate_lookup.index:
            continue

        s1_row = source1_lookup.loc[source1_id]
        candidate_row = candidate_lookup.loc[candidate_id]

        name_features = _name_features(
            s1_row,
            candidate_row,
        )

        address_features = _address_features(
            s1_row,
            candidate_row,
        )

        country_features = _country_features(
            s1_row,
            candidate_row,
        )

        evidence_features = _evidence_features(
            name_features,
            address_features,
            country_features,
        )

        row = {
            "source1_id": source1_id,
            "candidate_id": candidate_id,
            "candidate_source": candidate_source,
        }

        row.update(name_features)
        row.update(address_features)
        row.update(country_features)
        row.update(evidence_features)

        rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "build_pair_features",
]