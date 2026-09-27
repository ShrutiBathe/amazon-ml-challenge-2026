
"""
M2 - Blocking / Candidate Generation

Scalable blocking and candidate generation for the
Amazon ML Challenge 2026.

M1 provides normalized representations.
M2 uses those representations to generate a high-recall,
manageable candidate set for M3 matching.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

CANDIDATE_COLUMNS = [
    "source1_id",
    "candidate_id",
    "candidate_source",
    "blocking_strategy",
]


# ============================================================
# BLOCKING ENGINE
# ============================================================

class BlockingEngine:
    """
    Generate candidate pairs using multiple blocking strategies.

    The engine applies multiple blocking strategies and combines
    their candidate pairs while removing duplicate pairs.
    """

    def __init__(self) -> None:
        self.strategies: list[tuple[str, Callable]] = []

    def add_strategy(
        self,
        name: str,
        strategy: Callable,
    ) -> None:
        """Register a blocking strategy."""

        self.strategies.append((name, strategy))

    def generate_candidates(
        self,
        source1: pd.DataFrame,
        candidate_sources: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate candidate pairs from all registered strategies.

        Each strategy receives:
            source1
            candidate_sources

        The resulting candidate pairs are merged and deduplicated.
        """

        all_candidates: list[pd.DataFrame] = []

        for strategy_name, strategy in self.strategies:

            candidates = strategy(
                source1,
                candidate_sources,
            )

            if candidates is None or candidates.empty:
                continue

            candidates = candidates.copy()

            candidates["blocking_strategy"] = strategy_name

            all_candidates.append(candidates)

        if not all_candidates:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)

        result = pd.concat(
            all_candidates,
            ignore_index=True,
        )

        # Same pair may be discovered by multiple strategies.
        # Keep only one copy of the pair.
        result = result.drop_duplicates(
            subset=[
                "source1_id",
                "candidate_source",
                "candidate_id",
            ]
        ).reset_index(drop=True)

        return result


# ============================================================
# INDEX BUILDING
# ============================================================

def _build_index(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, list[str]]:
    """
    Build:

        blocking_value -> entity IDs
    """

    index: dict[str, list[str]] = defaultdict(list)

    if column not in dataframe.columns:
        return dict(index)

    for value, entity_id in zip(
        dataframe[column],
        dataframe["entity_id"],
    ):

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        index[value].append(str(entity_id))

    return dict(index)


def _build_composite_index(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[tuple[str, ...], list[str]]:
    """
    Build an index using multiple blocking fields.

    Example:

        (country, name_prefix_6)
            ->
        ["S2-123", "S2-456"]
    """

    index: dict[tuple[str, ...], list[str]] = defaultdict(list)

    required = set(columns) | {"entity_id"}

    if not required.issubset(dataframe.columns):
        return dict(index)

    for _, row in dataframe.iterrows():

        values: list[str] = []
        valid = True

        for column in columns:

            value = row[column]

            if pd.isna(value):
                valid = False
                break

            value = str(value).strip()

            if not value:
                valid = False
                break

            values.append(value)

        if not valid:
            continue

        index[tuple(values)].append(
            str(row["entity_id"])
        )

    return dict(index)


# ============================================================
# GENERIC SINGLE-COLUMN BLOCKING
# ============================================================

def _generate_from_column(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
    column: str,
) -> pd.DataFrame:
    """
    Generate candidates using exact equality on one M1 field.
    """

    records: list[dict] = []

    if column not in source1.columns:
        return pd.DataFrame(
            columns=[
                "source1_id",
                "candidate_id",
                "candidate_source",
            ]
        )

    for source_name, candidate_df in candidate_sources.items():

        candidate_index = _build_index(
            candidate_df,
            column,
        )

        for _, row in source1.iterrows():

            value = row[column]

            if pd.isna(value):
                continue

            value = str(value).strip()

            if not value:
                continue

            matches = candidate_index.get(
                value,
                [],
            )

            for candidate_id in matches:

                records.append(
                    {
                        "source1_id": str(
                            row["entity_id"]
                        ),
                        "candidate_id": candidate_id,
                        "candidate_source": source_name,
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
    )


# ============================================================
# M2 V3 GENERIC HELPER
# ============================================================

def _blocking_on_columns(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
    column: str,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    """
    Generic reusable blocking helper.

    This is used by V3 strategies such as:

        first_token_blocking
        address_first_token_blocking

    The strategy_name parameter is accepted for compatibility,
    but the BlockingEngine ultimately assigns the strategy name.
    """

    # strategy_name is intentionally not used here because
    # BlockingEngine adds the final strategy label.
    _ = strategy_name

    return _generate_from_column(
        source1,
        candidate_sources,
        column,
    )


# ============================================================
# ORIGINAL BLOCKING STRATEGIES
# ============================================================

def exact_name_blocking(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Block using exact compact business name."""

    return _generate_from_column(
        source1,
        candidate_sources,
        "name_compact",
    )


def name_prefix_blocking(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Block using M1's six-character name prefix."""

    return _generate_from_column(
        source1,
        candidate_sources,
        "name_prefix_6",
    )


def address_token_blocking(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Block using the first address token."""

    return _generate_from_column(
        source1,
        candidate_sources,
        "address_first_token",
    )


# ============================================================
# COUNTRY + NAME PREFIX
# ============================================================

def country_name_prefix_blocking(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Block using:

        country_normalized + name_prefix_6

    This reduces collisions where the same business-name
    prefix occurs in multiple countries.
    """

    records: list[dict] = []

    required_columns = {
        "country_normalized",
        "name_prefix_6",
    }

    if not required_columns.issubset(source1.columns):
        return pd.DataFrame()

    for source_name, candidate_df in candidate_sources.items():

        if not required_columns.issubset(candidate_df.columns):
            continue

        index = _build_composite_index(
            candidate_df,
            [
                "country_normalized",
                "name_prefix_6",
            ],
        )

        for _, row in source1.iterrows():

            country = row["country_normalized"]
            prefix = row["name_prefix_6"]

            if pd.isna(country) or pd.isna(prefix):
                continue

            country = str(country).strip()
            prefix = str(prefix).strip()

            if not country or not prefix:
                continue

            matches = index.get(
                (country, prefix),
                [],
            )

            for candidate_id in matches:

                records.append(
                    {
                        "source1_id": str(
                            row["entity_id"]
                        ),
                        "candidate_id": candidate_id,
                        "candidate_source": source_name,
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
    )


# ============================================================
# COUNTRY + ADDRESS TOKEN
# ============================================================

def country_address_token_blocking(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Block using:

        country_normalized + address_first_token
    """

    records: list[dict] = []

    required_columns = {
        "country_normalized",
        "address_first_token",
    }

    if not required_columns.issubset(source1.columns):
        return pd.DataFrame()

    for source_name, candidate_df in candidate_sources.items():

        if not required_columns.issubset(candidate_df.columns):
            continue

        index = _build_composite_index(
            candidate_df,
            [
                "country_normalized",
                "address_first_token",
            ],
        )

        for _, row in source1.iterrows():

            country = row["country_normalized"]
            address_token = row["address_first_token"]

            if (
                pd.isna(country)
                or pd.isna(address_token)
            ):
                continue

            country = str(country).strip()
            address_token = str(address_token).strip()

            if not country or not address_token:
                continue

            matches = index.get(
                (
                    country,
                    address_token,
                ),
                [],
            )

            for candidate_id in matches:

                records.append(
                    {
                        "source1_id": str(
                            row["entity_id"]
                        ),
                        "candidate_id": candidate_id,
                        "candidate_source": source_name,
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
    )


# ============================================================
# M2 V2 ENGINE
# ============================================================

def create_v2_engine() -> BlockingEngine:
    """
    Create M2 V2 blocking configuration.

    Includes baseline blocks plus source-aware composite blocks.
    """

    engine = BlockingEngine()

    engine.add_strategy(
        "exact_name",
        exact_name_blocking,
    )

    engine.add_strategy(
        "name_prefix",
        name_prefix_blocking,
    )

    engine.add_strategy(
        "address_token",
        address_token_blocking,
    )

    engine.add_strategy(
        "country_name_prefix",
        country_name_prefix_blocking,
    )

    engine.add_strategy(
        "country_address_token",
        country_address_token_blocking,
    )

    return engine


# ============================================================
# M2 V3 HIGH-RECALL STRATEGIES
# ============================================================

def first_token_blocking(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Block records using the first meaningful business-name token.
    """

    return _blocking_on_columns(
        source1,
        candidate_sources,
        column="name_first_token",
        strategy_name="first_token",
    )


def address_first_token_blocking(
    source1: pd.DataFrame,
    candidate_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Block records using the first address token.
    """

    return _blocking_on_columns(
        source1,
        candidate_sources,
        column="address_first_token",
        strategy_name="address_first_token",
    )
def country_address_overlap_blocking(source1, candidate_sources):
    """
    V4.1 blocking:
    Country + two informative shared address tokens.

    Common address tokens are ignored to prevent candidate explosion.
    """

    COMMON_ADDRESS_TOKENS = {
        "street", "st", "road", "rd", "avenue", "ave",
        "lane", "ln", "drive", "dr", "boulevard", "blvd",
        "floor", "fl", "house", "h", "block", "b",
        "unit", "shop", "plot", "flat", "building",
        "city", "district", "state",
        "india", "maharashtra", "karnataka", "gujarat",
        "delhi", "mumbai", "pune", "nashik",
        "hyderabad", "bangalore", "kolkata", "chennai",
        "road,", "street,", "avenue,"
    }

    records = []

    for source_name, source_df in candidate_sources.items():

        if source_df.empty:
            continue

        # --------------------------------------------------
        # Build country + token -> candidate IDs
        # --------------------------------------------------

        token_index = {}

        for _, row in source_df.iterrows():

            country = row.get("country_normalized")

            if pd.isna(country):
                continue

            address = str(
                row.get("address_normalized", "")
            ).strip().lower()

            if not address:
                continue

            tokens = {
                token.strip(".,#-/")
                for token in address.split()
                if len(token.strip(".,#-/")) >= 2
            }

            tokens -= COMMON_ADDRESS_TOKENS

            for token in tokens:

                key = (country, token)

                token_index.setdefault(
                    key,
                    set()
                ).add(row["entity_id"])

        # --------------------------------------------------
        # Remove extremely common tokens
        # --------------------------------------------------

        MAX_BLOCK_SIZE = 300

        for key in list(token_index.keys()):

            if len(token_index[key]) > MAX_BLOCK_SIZE:
                del token_index[key]

        # --------------------------------------------------
        # Generate candidates
        # --------------------------------------------------

        for _, row in source1.iterrows():

            source1_id = row["entity_id"]

            country = row.get("country_normalized")

            if pd.isna(country):
                continue

            address = str(
                row.get("address_normalized", "")
            ).strip().lower()

            if not address:
                continue

            tokens = {
                token.strip(".,#-/")
                for token in address.split()
                if len(token.strip(".,#-/")) >= 2
            }

            tokens -= COMMON_ADDRESS_TOKENS

            if len(tokens) < 2:
                continue

            candidate_counts = {}

            for token in tokens:

                key = (country, token)

                candidate_ids = token_index.get(key)

                if not candidate_ids:
                    continue

                for candidate_id in candidate_ids:

                    candidate_counts[candidate_id] = (
                        candidate_counts.get(candidate_id, 0) + 1
                    )

            # Require at least 2 informative shared tokens
            for candidate_id, shared_count in candidate_counts.items():

                if shared_count >= 2:

                    records.append(
                        {
                            "source1_id": source1_id,
                            "candidate_id": candidate_id,
                            "candidate_source": source_name,
                        }
                    )

    return pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
    )
    """
    Block records using country + at least 2 shared address tokens.

    This is designed to recover true matches where the address
    tokens are reordered or formatted differently between sources.
    """

    records = []

    for source_name, source_df in candidate_sources.items():
        if source_df.empty:
            continue

        # Build country -> token -> candidate IDs index
        index = {}

        for _, row in source_df.iterrows():
            country = row.get("country_normalized")

            if pd.isna(country):
                continue

            address = str(row.get("address_normalized", "")).strip().lower()

            if not address:
                continue

            tokens = {
                token
                for token in address.split()
                if len(token) >= 2
            }

            for token in tokens:
                key = (country, token)
                index.setdefault(key, []).append(row["entity_id"])

        for _, row in source1.iterrows():
            source1_id = row["entity_id"]
            country = row.get("country_normalized")

            if pd.isna(country):
                continue

            address = str(
                row.get("address_normalized", "")
            ).strip().lower()

            if not address:
                continue

            tokens = {
                token
                for token in address.split()
                if len(token) >= 2
            }

            if len(tokens) < 2:
                continue

            candidate_counts = {}

            for token in tokens:
                key = (country, token)

                for candidate_id in index.get(key, []):
                    candidate_counts[candidate_id] = (
                        candidate_counts.get(candidate_id, 0) + 1
                    )

            for candidate_id, shared_count in candidate_counts.items():
                if shared_count >= 2:
                    records.append(
                        {
                            "source1_id": source1_id,
                            "candidate_id": candidate_id,
                            "candidate_source": source_name,
                        }
                    )

    return pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
    )

# ============================================================
# M2 V3 ENGINE
# ============================================================

def create_v3_engine() -> BlockingEngine:
    """
    Create M2 V3 high-recall blocking configuration.

    V3 keeps baseline strategies and adds evidence-based
    first-token blocking discovered from ground-truth analysis.
    """

    engine = BlockingEngine()

    engine.add_strategy(
        "exact_name",
        exact_name_blocking,
    )

    engine.add_strategy(
        "name_prefix",
        name_prefix_blocking,
    )

    engine.add_strategy(
        "address_token",
        address_token_blocking,
    )

    engine.add_strategy(
        "first_token",
        first_token_blocking,
    )

    engine.add_strategy(
        "address_first_token",
        address_first_token_blocking,
    )

    return engine
def create_v4_engine() -> BlockingEngine:
    """
    Create M2 V4 blocking configuration.

    V4 adds country + address token overlap blocking
    to the V3 strategies.
    """

    engine = BlockingEngine()

    engine.add_strategy(
        "exact_name",
        exact_name_blocking,
    )

    engine.add_strategy(
        "name_prefix",
        name_prefix_blocking,
    )

    engine.add_strategy(
        "address_token",
        address_token_blocking,
    )

    engine.add_strategy(
        "first_token",
        first_token_blocking,
    )

    engine.add_strategy(
        "address_first_token",
        address_first_token_blocking,
    )

    engine.add_strategy(
        "country_address_overlap",
        country_address_overlap_blocking,
    )

    return engine

# ============================================================
# BASELINE ENGINE
# ============================================================

def create_baseline_engine() -> BlockingEngine:
    """
    Keep the original baseline engine available.

    Useful for A/B comparison:

        baseline vs V2 vs V3
    """

    engine = BlockingEngine()

    engine.add_strategy(
        "exact_name",
        exact_name_blocking,
    )

    engine.add_strategy(
        "name_prefix",
        name_prefix_blocking,
    )

    engine.add_strategy(
        "address_token",
        address_token_blocking,
    )

    return engine


# ============================================================
# QUICK MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("M2 BLOCKING MODULE")
    print("=" * 70)

    print()
    print("Available engines:")
    print("  - create_baseline_engine()")
    print("  - create_v2_engine()")
    print("  - create_v3_engine()")

    print()
    print("Available strategies:")
    print("  - exact_name")
    print("  - name_prefix")
    print("  - address_token")
    print("  - country_name_prefix")
    print("  - country_address_token")
    print("  - first_token")
    print("  - address_first_token")

    print()
    print("M2 blocking module loaded successfully.")

