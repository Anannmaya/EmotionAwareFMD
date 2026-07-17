from pathlib import Path
import re

import pandas as pd


RAW_DIR = Path("data/raw/liar2_finance")
OUTPUT_DIR = Path("data/interim")

FILES = {
    "train": RAW_DIR / "train.csv",
    "valid": RAW_DIR / "valid.csv",
    "test": RAW_DIR / "test.csv",
}

LABEL_MAP = {
    0: "pants_on_fire",
    1: "false",
    2: "barely_true",
    3: "half_true",
    4: "mostly_true",
    5: "true",
}

# Retain only the least ambiguous LIAR2 labels.
BINARY_LABEL_MAP = {
    0: 0,  # pants_on_fire -> misleading
    1: 0,  # false -> misleading
    4: 1,  # mostly_true -> genuine
    5: 1,  # true -> genuine
}

BINARY_LABEL_NAMES = {
    0: "misleading",
    1: "genuine",
}

# Subjects that can support financial relevance, but subject metadata alone
# is no longer enough to retain a claim.
CORE_FINANCIAL_SUBJECTS = {
    "bankruptcy",
    "banks",
    "budget",
    "business",
    "corporations",
    "debt",
    "deficit",
    "economy",
    "employment",
    "federal budget",
    "financial regulation",
    "gas prices",
    "government spending",
    "income",
    "inflation",
    "jobs",
    "labor",
    "minimum wage",
    "mortgages",
    "pensions",
    "recession",
    "social security",
    "state budget",
    "stimulus",
    "stock market",
    "taxes",
    "trade",
    "unemployment",
    "wages",
    "wealth",
    "workers",
}

# Subjects that commonly produced false positives in the two audits.
NON_TARGET_SUBJECTS = {
    "abortion",
    "baseball",
    "basketball",
    "campaign finance",
    "candidate biography",
    "crime",
    "criminal justice",
    "education",
    "elections",
    "entertainment",
    "football",
    "guns",
    "health care",
    "immigration",
    "legal issues",
    "religion",
    "sexuality",
    "sports",
}

# A match to one of these is strong enough to retain a claim on its own.
# Word boundaries prevent accidental substring matches such as "tax" in
# "attack" or "bank" inside another word.
HIGH_CONFIDENCE_PATTERNS = {
    "bankruptcy": r"\bbankrupt(?:cy|cies)?\b",
    "bailout": r"\bbailout(?:s)?\b",
    "bond_market": r"\bbond market(?:s)?\b",
    "budget_deficit": r"\bbudget deficit(?:s)?\b",
    "capital_gains": r"\bcapital gains?\b",
    "corporate_tax": r"\bcorporate tax(?:es)?\b",
    "cost_of_living": r"\bcost of living\b",
    "credit_rating": r"\bcredit rating(?:s)?\b",
    "economic_growth": r"\beconomic growth\b",
    "federal_budget": r"\bfederal budget\b",
    "federal_debt": r"\bfederal debt\b",
    "federal_deficit": r"\bfederal deficit\b",
    "financial_crisis": r"\bfinancial crisis\b",
    "gdp": r"\b(?:gdp|gross domestic product)\b",
    "government_debt": r"\bgovernment debt\b",
    "government_spending": r"\bgovernment spending\b",
    "income_tax": r"\bincome tax(?:es)?\b",
    "inflation": r"\binflation\b",
    "interest_rate": r"\binterest rates?\b",
    "minimum_wage": r"\bminimum wage\b",
    "mortgage": r"\bmortgages?\b",
    "national_debt": r"\bnational debt\b",
    "pension": r"\bpensions?\b",
    "property_tax": r"\bproperty tax(?:es)?\b",
    "public_debt": r"\bpublic debt\b",
    "public_spending": r"\bpublic spending\b",
    "recession": r"\brecession\b",
    "social_security": r"\bsocial security\b",
    "state_budget": r"\bstate budget\b",
    "stimulus": r"\bstimulus(?: package| bill| plan)?\b",
    "stock_market": r"\bstock market(?:s)?\b",
    "tax_revenue": r"\btax revenue\b",
    "trade_deficit": r"\btrade deficit(?:s)?\b",
    "unemployment_rate": r"\bunemployment rate\b",
}

# These are useful supporting signals. A claim needs:
#   1) a core financial subject plus at least one supporting signal, or
#   2) at least two different supporting signals.
SUPPORTING_PATTERNS = {
    "bank": r"\bbanks?\b",
    "bond": r"\bbonds?\b",
    "budget": r"\bbudgets?\b",
    "credit": r"\bcredit\b",
    "debt": r"\bdebts?\b",
    "deficit": r"\bdeficits?\b",
    "dividend": r"\bdividends?\b",
    "economic": r"\beconomic(?:ally)?\b",
    "economy": r"\beconom(?:y|ies)\b",
    "employment": r"\bemployment\b",
    "financial": r"\bfinancial(?:ly)?\b",
    "gas_price": r"\bgas prices?\b",
    "income": r"\bincome\b",
    "investment": r"\binvest(?:ment|ments|or|ors|ing)\b",
    "job": r"\bjobs?\b",
    "loan": r"\bloans?\b",
    "market": r"\bmarkets?\b",
    "profit": r"\bprofits?\b",
    "revenue": r"\brevenues?\b",
    "salary": r"\bsalar(?:y|ies)\b",
    "spending": r"\bspending\b",
    "stock": r"\bstocks?\b",
    "tariff": r"\btariffs?\b",
    "tax": r"\btax(?:es|ed|ing|ation)?\b",
    "trade": r"\btrade\b",
    "unemployment": r"\bunemployment\b",
    "wage": r"\bwages?\b",
}

# These patterns were frequent sources of off-topic matches in the audits.
# They do not automatically remove a claim if a high-confidence financial
# expression is also present.
OFF_TOPIC_PATTERNS = {
    "campaign_money": (
        r"\b(?:campaign contribution|campaign donation|campaign cash|"
        r"super pac|political donation|political contribution)s?\b"
    ),
    "endorsement": r"\bendorse(?:ment|ments|d|s)?\b",
    "quote_attribution": r"\b(?:said|quote|quoted|statement attributed)\b",
    "tax_return_disclosure": (
        r"\b(?:release|released|releasing|disclose|disclosed|"
        r"disclosing)\b.{0,30}\btax returns?\b"
    ),
    "sports_team": r"\b(?:football|basketball|baseball|sports?) team\b",
    "prison": r"\bprisons?\b",
}


def compile_patterns(patterns: dict[str, str]) -> dict[str, re.Pattern]:
    """Compile case-insensitive regex patterns once."""

    return {
        name: re.compile(pattern, flags=re.IGNORECASE)
        for name, pattern in patterns.items()
    }


HIGH_CONFIDENCE_REGEX = compile_patterns(HIGH_CONFIDENCE_PATTERNS)
SUPPORTING_REGEX = compile_patterns(SUPPORTING_PATTERNS)
OFF_TOPIC_REGEX = compile_patterns(OFF_TOPIC_PATTERNS)


def load_liar2() -> pd.DataFrame:
    """Load and combine all official LIAR2 splits."""

    frames = []

    for split_name, file_path in FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing raw file: {file_path}")

        frame = pd.read_csv(file_path)
        frame["original_split"] = split_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def normalise_statement(text: str) -> str:
    """Create a conservative normalised form for duplicate detection."""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_subjects(subject: object) -> set[str]:
    """Split LIAR2 semicolon-separated subject metadata."""

    if pd.isna(subject):
        return set()

    return {
        item.strip().lower()
        for item in str(subject).split(";")
        if item.strip()
    }


def match_patterns(
    text: str,
    patterns: dict[str, re.Pattern],
) -> set[str]:
    """Return the names of all regex patterns matched in text."""

    return {
        name
        for name, pattern in patterns.items()
        if pattern.search(text)
    }


def find_financial_matches(
    row: pd.Series,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return matched subjects and text-pattern groups."""

    subjects = split_subjects(row.get("subject"))
    statement = str(row.get("statement", ""))

    matched_subjects = subjects & CORE_FINANCIAL_SUBJECTS
    high_matches = match_patterns(statement, HIGH_CONFIDENCE_REGEX)
    supporting_matches = match_patterns(statement, SUPPORTING_REGEX)
    off_topic_matches = match_patterns(statement, OFF_TOPIC_REGEX)

    return (
        matched_subjects,
        high_matches,
        supporting_matches,
        off_topic_matches,
    )


def is_financial_candidate(row: pd.Series) -> bool:
    """Apply the final high-precision financial relevance rule."""

    subjects = split_subjects(row.get("subject"))

    (
        matched_subjects,
        high_matches,
        supporting_matches,
        off_topic_matches,
    ) = find_financial_matches(row)

    # A strong financial expression is sufficient even if subject metadata
    # is missing or noisy.
    if high_matches:
        return True

    # Campaign-finance/election rows should not pass merely because they
    # mention one generic monetary term.
    has_campaign_or_election_subject = bool(
        subjects & {"campaign finance", "elections"}
    )

    if has_campaign_or_election_subject and len(supporting_matches) < 2:
        return False

    # Reject common off-topic formulations unless the statement contains
    # at least two distinct financial signals and has financial metadata.
    if off_topic_matches:
        return bool(matched_subjects) and len(supporting_matches) >= 2

    # Standard route: financial metadata must be supported by wording in
    # the actual claim.
    if matched_subjects and supporting_matches:
        return True

    # Text-only route: require at least two different financial concepts.
    if len(supporting_matches) >= 2:
        return True

    # Claims whose subjects are exclusively non-target are rejected.
    if subjects and subjects.issubset(NON_TARGET_SUBJECTS):
        return False

    return False


def add_filter_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Record exactly why each row passed the filter."""

    dataframe = dataframe.copy()

    matched_subject_values = []
    high_match_values = []
    supporting_match_values = []
    off_topic_match_values = []
    match_sources = []

    for _, row in dataframe.iterrows():
        (
            matched_subjects,
            high_matches,
            supporting_matches,
            off_topic_matches,
        ) = find_financial_matches(row)

        matched_subject_values.append(";".join(sorted(matched_subjects)))
        high_match_values.append(";".join(sorted(high_matches)))
        supporting_match_values.append(";".join(sorted(supporting_matches)))
        off_topic_match_values.append(";".join(sorted(off_topic_matches)))

        if high_matches and matched_subjects:
            source = "high_confidence_and_subject"
        elif high_matches:
            source = "high_confidence_text"
        elif matched_subjects and supporting_matches:
            source = "subject_and_supporting_text"
        elif len(supporting_matches) >= 2:
            source = "multiple_text_signals"
        else:
            source = "other"

        match_sources.append(source)

    dataframe["matched_financial_subjects"] = matched_subject_values
    dataframe["matched_high_confidence_terms"] = high_match_values
    dataframe["matched_supporting_terms"] = supporting_match_values
    dataframe["matched_off_topic_patterns"] = off_topic_match_values
    dataframe["financial_match_source"] = match_sources

    return dataframe


def print_summary(name: str, dataframe: pd.DataFrame) -> None:
    """Print a concise dataset summary."""

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows: {len(dataframe):,}")
    print(f"Unique statements: {dataframe['statement'].nunique():,}")

    if "label_name" in dataframe.columns:
        print("\nOriginal label distribution:")
        print(dataframe["label_name"].value_counts().to_string())

    if "binary_label_name" in dataframe.columns:
        print("\nBinary label distribution:")
        print(dataframe["binary_label_name"].value_counts().to_string())

    if "financial_match_source" in dataframe.columns:
        print("\nFinancial match source:")
        print(dataframe["financial_match_source"].value_counts().to_string())

    print("\nMost frequent subject combinations:")
    print(
        dataframe["subject"]
        .fillna("[missing]")
        .value_counts()
        .head(20)
        .to_string()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_liar2()
    dataframe["label_name"] = dataframe["label"].map(LABEL_MAP)

    print_summary("FULL LIAR2 DATASET", dataframe)

    candidates = dataframe[
        dataframe.apply(is_financial_candidate, axis=1)
    ].copy()

    candidates = add_filter_metadata(candidates)

    print_summary(
        "HIGH-PRECISION FINANCE/ECONOMICS CANDIDATES — ALL LABELS",
        candidates,
    )

    # Retain only unambiguous binary labels.
    candidates = candidates[
        candidates["label"].isin(BINARY_LABEL_MAP)
    ].copy()

    candidates["binary_label"] = (
        candidates["label"]
        .map(BINARY_LABEL_MAP)
        .astype(int)
    )

    candidates["binary_label_name"] = (
        candidates["binary_label"]
        .map(BINARY_LABEL_NAMES)
    )

    # Clean statement text conservatively.
    candidates["statement"] = (
        candidates["statement"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    candidates = candidates[
        candidates["statement"].str.len() > 0
    ].copy()

    candidates["word_count"] = (
        candidates["statement"]
        .str.split()
        .str.len()
    )

    before_short_filter = len(candidates)
    candidates = candidates[
        candidates["word_count"] >= 5
    ].copy()

    print(
        "\nRemoved statements shorter than five words: "
        f"{before_short_filter - len(candidates):,}"
    )

    # Remove exact and punctuation-only duplicate variants.
    candidates["statement_normalised"] = (
        candidates["statement"]
        .map(normalise_statement)
    )

    before_deduplication = len(candidates)

    candidates = candidates.drop_duplicates(
        subset="statement_normalised",
        keep="first",
    ).copy()

    print(
        "Removed duplicate or near-identical statements: "
        f"{before_deduplication - len(candidates):,}"
    )

    candidates = candidates.reset_index(drop=True)

    print_summary(
        "CLEAN HIGH-PRECISION BINARY FINANCE/ECONOMICS CANDIDATES",
        candidates,
    )

    print("\nStatement-length statistics:")
    print(
        candidates
        .groupby("binary_label_name")["word_count"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
        .to_string()
    )

    candidate_path = (
        OUTPUT_DIR / "liar2_finance_candidates_high_precision.csv"
    )
    candidates.to_csv(candidate_path, index=False)

    # Fresh balanced audit sample with a new random seed.
    audit_parts = []

    for label_value in [0, 1]:
        subset = candidates[
            candidates["binary_label"] == label_value
        ]

        sample_size = min(50, len(subset))

        audit_parts.append(
            subset.sample(
                n=sample_size,
                random_state=126,
            )
        )

    audit_sample = (
        pd.concat(audit_parts, ignore_index=True)
        .sample(frac=1, random_state=126)
        .reset_index(drop=True)
    )

    audit_sample["financially_relevant"] = ""
    audit_sample["standalone_claim"] = ""
    audit_sample["audit_notes"] = ""

    audit_columns = [
        "id",
        "statement",
        "subject",
        "matched_financial_subjects",
        "matched_high_confidence_terms",
        "matched_supporting_terms",
        "matched_off_topic_patterns",
        "financial_match_source",
        "binary_label",
        "binary_label_name",
        "speaker",
        "date",
        "context",
        "original_split",
        "word_count",
        "financially_relevant",
        "standalone_claim",
        "audit_notes",
    ]

    audit_path = (
        OUTPUT_DIR / "liar2_finance_audit_sample_v3.csv"
    )

    audit_sample[audit_columns].to_csv(
        audit_path,
        index=False,
    )

    print("\nSaved files:")
    print(candidate_path)
    print(audit_path)


if __name__ == "__main__":
    main()
