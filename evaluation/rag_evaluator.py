# ============================================================
# ResearchPilot — FINAL EVIDENCE-LEVEL RETRIEVAL EVALUATOR
#
# Evaluates:
#
# Qwen3-Embedding-8B
#       ↓
# Dense Top-CANDIDATE_K
#       ↓
# Qwen3-Reranker-4B
#       ↓
# Final Top-FINAL_K
#
# This evaluator intentionally measures RETRIEVAL ONLY.
#
# It does NOT:
# - generate LLM answers
# - run calculator tools
# - run source-aware company filtering used by answer generation
#
# Those are measured separately by the final answer evaluator.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import html
import importlib
import json
import math
import re
import sys
import time

from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px


# ============================================================
# PROJECT DIRECTORY
#
# In a normal .py file:
#     uses the directory containing this file.
#
# In Kaggle / Jupyter:
#     uses the current working directory.
#
# Therefore, run the notebook/cell from the ResearchPilot
# project directory.
# ============================================================

if "__file__" in globals():

    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parent
    )

else:

    BASE_DIR = (
        Path.cwd()
        .resolve()
    )


print(
    f"Project directory: {BASE_DIR}"
)


# ============================================================
# PATHS
# ============================================================

RETRIEVAL_FILE = (
    BASE_DIR
    / "retrieval.py"
)

GOLDEN_DATASET_PATH = (
    BASE_DIR
    / "golden_dataset.jsonl"
)

RESULTS_CSV_PATH = (
    BASE_DIR
    / "final_rag_eval_results.csv"
)

SUMMARY_CSV_PATH = (
    BASE_DIR
    / "final_rag_eval_summary.csv"
)


# ============================================================
# EVALUATION CONFIG
# ============================================================

EVIDENCE_MATCH_THRESHOLD = 0.80

# Lower these if CUDA OOM occurs.
QUERY_BATCH_SIZE = 16

# If None, use the production reranker batch size from
# retrieval.py / config.py.
EVALUATOR_RERANK_BATCH_SIZE = None


# ============================================================
# VERIFY PROJECT FILES
# ============================================================

if not RETRIEVAL_FILE.exists():

    raise FileNotFoundError(
        f"Retrieval file not found:\n"
        f"{RETRIEVAL_FILE}"
    )


if not GOLDEN_DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Golden dataset not found:\n"
        f"{GOLDEN_DATASET_PATH}"
    )


# ============================================================
# IMPORT retrieval.py
#
# Using a normal Python import is preferable now that
# retrieval.py is a proper packaged Python file.
#
# No more stripping !pip / notebook magic lines.
# ============================================================

if str(BASE_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BASE_DIR)
    )


importlib.invalidate_caches()


RR = importlib.import_module(
    "retrieval"
)


print(
    "✅ retrieval.py imported"
)


# ============================================================
# VERIFY REQUIRED RETRIEVAL COMPONENTS
# ============================================================

REQUIRED_ATTRIBUTES = [

    "embedding_model",
    "reranker_model",
    "vectorstore",
    "CANDIDATE_K",
    "FINAL_K",
]


missing_attributes = [

    name

    for name
    in REQUIRED_ATTRIBUTES

    if not hasattr(
        RR,
        name
    )
]


if missing_attributes:

    raise AttributeError(

        "retrieval.py is missing required attributes:\n"

        + "\n".join(
            missing_attributes
        )
    )


embedding_model = (
    RR.embedding_model
)

reranker_model = (
    RR.reranker_model
)

vectorstore = (
    RR.vectorstore
)


CANDIDATE_K = int(
    RR.CANDIDATE_K
)

FINAL_K = int(
    RR.FINAL_K
)


if FINAL_K > CANDIDATE_K:

    raise ValueError(
        f"FINAL_K ({FINAL_K}) cannot be greater "
        f"than CANDIDATE_K ({CANDIDATE_K})."
    )


# ============================================================
# USE PRODUCTION RERANKER BATCH SIZE WHEN AVAILABLE
# ============================================================

production_rerank_batch_size = int(

    getattr(
        RR,
        "RERANKER_BATCH_SIZE",
        20
    )
)


RERANK_BATCH_SIZE = (

    production_rerank_batch_size

    if EVALUATOR_RERANK_BATCH_SIZE
    is None

    else int(
        EVALUATOR_RERANK_BATCH_SIZE
    )
)


print(
    f"Candidate K:       {CANDIDATE_K}"
)

print(
    f"Final K:           {FINAL_K}"
)

print(
    f"Query batch size:  {QUERY_BATCH_SIZE}"
)

print(
    f"Rerank batch size: {RERANK_BATCH_SIZE}"
)


# ============================================================
# VERIFY VECTOR DATABASE
#
# The evaluator does NOT define its own database path.
# retrieval.py / config.py owns that configuration.
# ============================================================

retrieval_db_path = getattr(
    RR,
    "DB_NAME",
    getattr(
        RR,
        "db_name",
        None
    )
)


if retrieval_db_path:

    retrieval_db_path = Path(
        retrieval_db_path
    )


    if not retrieval_db_path.exists():

        raise FileNotFoundError(

            "Vector database configured by retrieval.py "
            "does not exist:\n"

            f"{retrieval_db_path}"
        )


    print(
        "✅ Vector DB:"
    )

    print(
        retrieval_db_path
    )


else:

    print(
        "ℹ️ retrieval.py did not expose DB_NAME/db_name; "
        "vectorstore itself loaded successfully."
    )


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

def load_golden_dataset(path):

    path = Path(path)


    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        dataset = [

            json.loads(line)

            for line in f

            if line.strip()
        ]


    if not dataset:

        raise ValueError(
            "Golden dataset is empty."
        )


    ids = []


    for item in dataset:

        item_id = item.get(
            "id"
        )

        ids.append(
            item_id
        )


        if not item_id:

            raise ValueError(
                "A golden-dataset item is missing its ID."
            )


        if not item.get(
            "question"
        ):

            raise ValueError(
                f"Missing question: {item_id}"
            )


        evidence_items = item.get(
            "evidence"
        )


        if not evidence_items:

            raise ValueError(
                f"No evidence: {item_id}"
            )


        for evidence in evidence_items:

            if not evidence.get(
                "document"
            ):

                raise ValueError(
                    f"Evidence document missing: {item_id}"
                )


            if not evidence.get(
                "text"
            ):

                raise ValueError(
                    f"Evidence text missing: {item_id}"
                )


    # --------------------------------------------------------
    # Duplicate-ID protection
    # --------------------------------------------------------

    duplicate_ids = [

        item_id

        for item_id, count
        in Counter(ids).items()

        if count > 1
    ]


    if duplicate_ids:

        raise ValueError(

            "Duplicate golden-dataset IDs found:\n"

            + "\n".join(
                duplicate_ids
            )
        )


    return dataset


golden_dataset = (
    load_golden_dataset(
        GOLDEN_DATASET_PATH
    )
)


print(
    f"✅ Loaded {len(golden_dataset)} "
    "evidence-grounded questions"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_source(source):

    if source is None:

        return ""


    source = (

        Path(
            str(source)
        )
        .name
        .lower()
        .strip()
    )


    for suffix in (
        ".pdf",
        ".md",
    ):

        if source.endswith(
            suffix
        ):

            source = source[
                :-len(suffix)
            ]


    return source


def normalize_text(text):

    if text is None:

        return ""


    text = html.unescape(
        str(text)
    )


    # US\$ → US$
    text = text.replace(
        "\\$",
        "$"
    )


    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )


    # Remove Markdown images
    text = re.sub(
        r"!\[[^\]]*\]\([^)]*\)",
        " ",
        text
    )


    # Keep Markdown link text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]*\)",
        r"\1",
        text
    )


    text = (
        text
        .replace(
            "**",
            " "
        )
        .replace(
            "__",
            " "
        )
        .replace(
            "`",
            " "
        )
    )


    # Currency normalization
    text = text.replace(
        "₹",
        " rs "
    )


    text = (
        text
        .replace(
            "US$",
            " usd "
        )
        .replace(
            "us$",
            " usd "
        )
    )


    # 1,78,650 → 178650
    text = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        text
    )


    text = text.lower()


    text = re.sub(
        r"[^a-z0-9.%+\-/]+",
        " ",
        text
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    return text


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {

    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",

    "year",
    "fy",
    "fiscal",

    "company",

    "crore",
    "crores",

    "report",
    "reported",
}


# ============================================================
# TOKENIZATION HELPERS
# ============================================================

def informative_tokens(text):

    tokens = re.findall(

        r"[a-z]+(?:-[a-z]+)*|\d+(?:\.\d+)?%?",

        normalize_text(
            text
        )
    )


    return [

        token

        for token in tokens

        if (
            token
            not in STOPWORDS

            and

            len(token) > 1
        )
    ]


def numeric_tokens(text):

    return re.findall(

        r"\d+(?:\.\d+)?%?",

        normalize_text(
            text
        )
    )


# ============================================================
# EVIDENCE MATCHING
# ============================================================

def evidence_match_score(
    evidence_text,
    chunk_text
):

    evidence_norm = normalize_text(
        evidence_text
    )

    chunk_norm = normalize_text(
        chunk_text
    )


    if (
        not evidence_norm
        or
        not chunk_norm
    ):

        return 0.0


    # --------------------------------------------------------
    # Perfect normalized substring match
    # --------------------------------------------------------

    if evidence_norm in chunk_norm:

        return 1.0


    evidence_tokens = informative_tokens(
        evidence_text
    )


    if not evidence_tokens:

        return 0.0


    evidence_counts = Counter(
        evidence_tokens
    )

    chunk_counts = Counter(
        informative_tokens(
            chunk_text
        )
    )


    matched = sum(

        min(
            count,
            chunk_counts[token]
        )

        for token, count
        in evidence_counts.items()
    )


    token_recall = (

        matched
        /
        sum(
            evidence_counts.values()
        )
    )


    # --------------------------------------------------------
    # Numeric agreement
    # --------------------------------------------------------

    evidence_numbers = numeric_tokens(
        evidence_text
    )


    if evidence_numbers:

        chunk_numbers = set(
            numeric_tokens(
                chunk_text
            )
        )


        number_recall = (

            sum(

                number
                in chunk_numbers

                for number
                in evidence_numbers
            )

            /
            len(
                evidence_numbers
            )
        )


        score = (

            0.75
            * token_recall

            +

            0.25
            * number_recall
        )


    else:

        score = token_recall


    return float(

        min(
            max(
                score,
                0.0
            ),
            1.0
        )
    )


# ============================================================
# BATCH DENSE CANDIDATE RETRIEVAL
#
# This is still the same production retrieval:
#
# encode(question, prompt_name="query")
# → Chroma Top CANDIDATE_K
#
# We simply batch all evaluation questions for efficiency.
# ============================================================

def retrieve_all_candidates():

    questions = [

        item[
            "question"
        ]

        for item
        in golden_dataset
    ]


    print(
        "\nEmbedding all evaluation questions..."
    )


    query_embeddings = embedding_model.encode(

        questions,

        prompt_name=
            "query",

        batch_size=
            QUERY_BATCH_SIZE,

        show_progress_bar=
            True,

        convert_to_numpy=
            True,
    )


    query_embeddings = np.asarray(
        query_embeddings
    )


    if len(
        query_embeddings
    ) != len(
        questions
    ):

        raise RuntimeError(
            "Embedding count does not match question count."
        )


    print(

        "\nQuerying Chroma for "
        f"Top-{CANDIDATE_K} candidates..."
    )


    results = vectorstore.query(

        query_embeddings=
            query_embeddings.tolist(),

        n_results=
            CANDIDATE_K,

        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


    all_candidates = []


    for question_index in range(
        len(
            questions
        )
    ):

        documents = (

            results[
                "documents"
            ][question_index]

            or []
        )


        metadatas = (

            results[
                "metadatas"
            ][question_index]

            or [
                {}
                for _ in documents
            ]
        )


        distances = (

            results[
                "distances"
            ][question_index]

            or [
                None
                for _ in documents
            ]
        )


        if not (

            len(
                documents
            )

            ==

            len(
                metadatas
            )

            ==

            len(
                distances
            )

        ):

            raise RuntimeError(

                "Chroma returned mismatched document, "
                "metadata, or distance lengths."
            )


        candidates = []


        for rank, (
            text,
            metadata,
            distance,

        ) in enumerate(

            zip(
                documents,
                metadatas,
                distances
            ),

            start=1
        ):

            metadata = (
                metadata
                or {}
            )


            candidates.append(
                {

                    "text":
                        text or "",

                    "metadata":
                        metadata,

                    "source":
                        normalize_source(
                            metadata.get(
                                "source",
                                ""
                            )
                        ),

                    "dense_rank":
                        rank,

                    "distance":
                        (
                            float(
                                distance
                            )

                            if distance
                            is not None

                            else None
                        ),

                    "rerank_score":
                        None,
                }
            )


        all_candidates.append(
            candidates
        )


    return all_candidates


# ============================================================
# BATCH RERANK ALL QUERY-PASSAGE PAIRS
#
# Example:
#
# 78 questions × 40 candidates = 3,120 pairs
#
# One predict() call with internal batching is much faster
# than 78 separate model calls while preserving the same
# query-passage scoring method.
# ============================================================

def rerank_all_candidates(
    all_candidates
):

    all_pairs = []

    question_boundaries = []

    start = 0


    for item, candidates in zip(
        golden_dataset,
        all_candidates
    ):

        question = item[
            "question"
        ]


        pairs = [

            (
                question,
                candidate[
                    "text"
                ]
            )

            for candidate
            in candidates
        ]


        all_pairs.extend(
            pairs
        )


        end = (
            start
            +
            len(
                pairs
            )
        )


        question_boundaries.append(
            (
                start,
                end
            )
        )


        start = end


    print(

        "\nReranking "
        f"{len(all_pairs):,} "
        "query-passage pairs..."
    )


    scores = reranker_model.predict(

        all_pairs,

        batch_size=
            RERANK_BATCH_SIZE,

        show_progress_bar=
            True,
    )


    scores = np.asarray(
        scores
    ).reshape(
        -1
    )


    if len(
        scores
    ) != len(
        all_pairs
    ):

        raise RuntimeError(

            "Reranker score count does not "
            "match query-passage pair count."
        )


    all_reranked = []


    for candidates, (
        start,
        end

    ) in zip(

        all_candidates,
        question_boundaries
    ):

        question_scores = scores[
            start:end
        ]


        reranked = []


        for candidate, score in zip(
            candidates,
            question_scores
        ):

            candidate_copy = dict(
                candidate
            )


            candidate_copy[
                "rerank_score"
            ] = float(
                score
            )


            reranked.append(
                candidate_copy
            )


        reranked.sort(

            key=lambda item:
                item[
                    "rerank_score"
                ],

            reverse=True
        )


        for rank, candidate in enumerate(
            reranked,
            start=1
        ):

            candidate[
                "rerank_rank"
            ] = rank


        all_reranked.append(
            reranked
        )


    return all_reranked


# ============================================================
# EVALUATE ONE FINAL TOP-K LIST
# ============================================================

def evaluate_ranked_list(
    item,
    ranked_items
):

    ranked_items = ranked_items[
        :FINAL_K
    ]


    evidence_items = item[
        "evidence"
    ]


    evidence_rows = []


    for evidence_index, evidence in enumerate(
        evidence_items
    ):

        expected_source = normalize_source(

            evidence.get(
                "document",
                ""
            )
        )


        best_score = 0.0
        best_rank = None
        first_match_rank = None


        for rank, candidate in enumerate(
            ranked_items,
            start=1
        ):

            # ------------------------------------------------
            # Evidence only matches passages from the
            # correct source document.
            # ------------------------------------------------

            if (
                candidate[
                    "source"
                ]
                !=
                expected_source
            ):

                continue


            score = evidence_match_score(

                evidence.get(
                    "text",
                    ""
                ),

                candidate[
                    "text"
                ]
            )


            if score > best_score:

                best_score = score
                best_rank = rank


            if (
                first_match_rank
                is None

                and

                score
                >=
                EVIDENCE_MATCH_THRESHOLD
            ):

                first_match_rank = rank


        matched = (
            first_match_rank
            is not None
        )


        evidence_rows.append(
            {

                "evidence_id":
                    evidence.get(
                        "evidence_id",
                        f"e{evidence_index + 1}"
                    ),

                "source":
                    expected_source,

                "claim":
                    evidence.get(
                        "claim",
                        ""
                    ),

                "matched":
                    matched,

                "first_match_rank":
                    first_match_rank,

                "best_rank":
                    best_rank,

                "best_score":
                    best_score,
            }
        )


    # ========================================================
    # HIT / EVIDENCE RECALL / FULL COVERAGE
    # ========================================================

    total_evidence = len(
        evidence_items
    )


    matched_count = sum(

        row[
            "matched"
        ]

        for row
        in evidence_rows
    )


    hit = float(
        matched_count > 0
    )


    evidence_recall = (

        matched_count
        /
        total_evidence

        if total_evidence

        else 0.0
    )


    full_coverage = float(

        total_evidence > 0

        and

        matched_count
        ==
        total_evidence
    )


    # ========================================================
    # MRR
    # ========================================================

    matched_ranks = [

        row[
            "first_match_rank"
        ]

        for row
        in evidence_rows

        if row[
            "first_match_rank"
        ]
        is not None
    ]


    first_evidence_rank = (

        min(
            matched_ranks
        )

        if matched_ranks

        else None
    )


    reciprocal_rank = (

        1.0
        /
        first_evidence_rank

        if first_evidence_rank

        else 0.0
    )


    # ========================================================
    # PROJECT nDCG
    #
    # IMPORTANT:
    #
    # This intentionally preserves the project's historical
    # nDCG definition for experiment comparability.
    #
    # It measures the ordering of evidence that WAS retrieved.
    # Evidence Recall / Full Coverage measure missing evidence.
    #
    # Do not describe this as textbook IR nDCG without this
    # qualification.
    # ========================================================

    rank_new_evidence = [

        0
        for _ in ranked_items
    ]


    for row in evidence_rows:

        if (
            row[
                "first_match_rank"
            ]
            is not None
        ):

            rank_new_evidence[
                row[
                    "first_match_rank"
                ]
                - 1
            ] += 1


    relevance = [

        1
        if gain > 0
        else 0

        for gain
        in rank_new_evidence
    ]


    dcg = sum(

        rel
        /
        math.log2(
            rank + 1
        )

        for rank, rel
        in enumerate(
            relevance,
            start=1
        )
    )


    relevant_retrieved_ranks = sum(
        relevance
    )


    idcg = sum(

        1
        /
        math.log2(
            rank + 1
        )

        for rank
        in range(
            1,
            relevant_retrieved_ranks + 1
        )
    )


    ndcg = (

        dcg
        /
        idcg

        if idcg

        else 0.0
    )


    ndcg = float(

        min(
            max(
                ndcg,
                0.0
            ),
            1.0
        )
    )


    return {

        "MRR":
            reciprocal_rank,

        "Hit":
            hit,

        "Evidence Recall":
            evidence_recall,

        "Full Coverage":
            full_coverage,

        "nDCG":
            ndcg,

        "First Evidence Rank":
            first_evidence_rank,

        "Matched Evidence":
            matched_count,

        "Evidence Rows":
            evidence_rows,
    }


# ============================================================
# DISPLAY / DEBUG HELPERS
# ============================================================

def evidence_status_string(
    metric_result
):

    parts = []


    for row in metric_result[
        "Evidence Rows"
    ]:

        if row[
            "matched"
        ]:

            parts.append(

                f"{row['evidence_id']}"
                f"@{row['first_match_rank']} "
                f"({row['best_score']:.2f})"
            )


        else:

            parts.append(

                f"{row['evidence_id']} "
                f"NOT FOUND "
                f"(best={row['best_score']:.2f})"
            )


    return " | ".join(
        parts
    )


def source_string(
    candidates
):

    return " | ".join(

        candidate[
            "source"
        ]

        for candidate
        in candidates[
            :FINAL_K
        ]
    )


def score_string(
    candidates
):

    return " | ".join(

        f"{candidate['rerank_score']:.3f}"

        for candidate
        in candidates[
            :FINAL_K
        ]
    )


# ============================================================
# FULL PIPELINE EVALUATION
# ============================================================

def evaluate_pipeline(
    progress=gr.Progress()
):

    start_time = time.time()


    # ========================================================
    # RETRIEVAL
    # ========================================================

    progress(
        0.05,
        desc=(
            "Embedding queries and "
            "retrieving candidates..."
        )
    )


    all_candidates = (
        retrieve_all_candidates()
    )


    # ========================================================
    # RERANKING
    # ========================================================

    progress(
        0.20,
        desc=(
            f"Reranking Top-{CANDIDATE_K} "
            "candidate pools..."
        )
    )


    all_reranked = (
        rerank_all_candidates(
            all_candidates
        )
    )


    # ========================================================
    # FINAL TOP-K EVIDENCE SCORING
    # ========================================================

    progress(
        0.85,
        desc=(
            "Scoring evidence retrieval..."
        )
    )


    rows = []


    for index, (
        item,
        candidates,
        reranked

    ) in enumerate(

        zip(
            golden_dataset,
            all_candidates,
            all_reranked
        )
    ):

        final_top_k = reranked[
            :FINAL_K
        ]


        metrics = evaluate_ranked_list(
            item,
            final_top_k
        )


        rows.append(
            {

                "ID":
                    item.get(
                        "id",
                        ""
                    ),

                "Company":
                    item.get(
                        "company",
                        "unknown"
                    ),

                "Category":
                    item.get(
                        "category",
                        "unknown"
                    ),

                "Difficulty":
                    item.get(
                        "difficulty",
                        "unknown"
                    ),

                "Hops":
                    item.get(
                        "hops",
                        1
                    ),

                "Question":
                    item[
                        "question"
                    ],

                "Expected Answer":
                    item.get(
                        "expected_answer",
                        ""
                    ),

                "Expected Evidence":
                    len(
                        item[
                            "evidence"
                        ]
                    ),

                "Matched Evidence":
                    metrics[
                        "Matched Evidence"
                    ],

                "First Evidence Rank":
                    (
                        metrics[
                            "First Evidence Rank"
                        ]

                        if metrics[
                            "First Evidence Rank"
                        ]
                        is not None

                        else
                        "Not Retrieved"
                    ),

                "MRR":
                    metrics[
                        "MRR"
                    ],

                "Hit":
                    metrics[
                        "Hit"
                    ],

                "Evidence Recall":
                    metrics[
                        "Evidence Recall"
                    ],

                "Full Coverage":
                    metrics[
                        "Full Coverage"
                    ],

                "nDCG":
                    metrics[
                        "nDCG"
                    ],

                "Evidence Matches":
                    evidence_status_string(
                        metrics
                    ),

                f"Top {FINAL_K} Sources":
                    source_string(
                        final_top_k
                    ),

                f"Top {FINAL_K} Reranker Scores":
                    score_string(
                        final_top_k
                    ),
            }
        )


        progress(

            0.85
            +
            0.14
            *
            (
                (index + 1)
                /
                len(
                    golden_dataset
                )
            ),

            desc=(

                f"Scoring "
                f"{index + 1}/"
                f"{len(golden_dataset)}"
            )
        )


    results_df = pd.DataFrame(
        rows
    )


    elapsed = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    overall = {

        "Questions":
            len(
                results_df
            ),

        "Candidate K":
            CANDIDATE_K,

        "Final K":
            FINAL_K,

        "Evidence Match Threshold":
            EVIDENCE_MATCH_THRESHOLD,

        "MRR":
            float(
                results_df[
                    "MRR"
                ].mean()
            ),

        "Hit":
            float(
                results_df[
                    "Hit"
                ].mean()
            ),

        "Evidence Recall":
            float(
                results_df[
                    "Evidence Recall"
                ].mean()
            ),

        "Full Coverage":
            float(
                results_df[
                    "Full Coverage"
                ].mean()
            ),

        "nDCG":
            float(
                results_df[
                    "nDCG"
                ].mean()
            ),

        "Runtime Seconds":
            elapsed,
    }


    summary_df = pd.DataFrame(
        [
            overall
        ]
    )


    # ========================================================
    # GROUPED RESULTS
    # ========================================================

    def grouped(
        column
    ):

        return (

            results_df

            .groupby(
                column
            )

            .agg(

                Questions=(
                    "ID",
                    "count"
                ),

                MRR=(
                    "MRR",
                    "mean"
                ),

                Hit=(
                    "Hit",
                    "mean"
                ),

                Evidence_Recall=(
                    "Evidence Recall",
                    "mean"
                ),

                Full_Coverage=(
                    "Full Coverage",
                    "mean"
                ),

                nDCG=(
                    "nDCG",
                    "mean"
                )
            )

            .reset_index()
        )


    category_df = grouped(
        "Category"
    )


    difficulty_df = grouped(
        "Difficulty"
    )


    hops_df = (

        grouped(
            "Hops"
        )

        .sort_values(
            "Hops"
        )
    )


    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    results_df.to_csv(
        RESULTS_CSV_PATH,
        index=False
    )


    summary_df.to_csv(
        SUMMARY_CSV_PATH,
        index=False
    )


    print(
        "\n✅ Evaluation complete"
    )

    print(
        f"Detailed results: {RESULTS_CSV_PATH}"
    )

    print(
        f"Summary:          {SUMMARY_CSV_PATH}"
    )


    return (

        overall,

        results_df,

        summary_df,

        category_df,

        difficulty_df,

        hops_df
    )


# ============================================================
# CHARTS
# ============================================================

METRICS = [

    "MRR",
    "Hit",
    "Evidence Recall",
    "Full Coverage",
    "nDCG",
]


def overall_plot(
    overall
):

    df = pd.DataFrame(
        {

            "Metric":
                METRICS,

            "Score": [

                overall[
                    metric
                ]

                for metric
                in METRICS
            ]
        }
    )


    fig = px.bar(

        df,

        x="Metric",

        y="Score",

        text="Score",

        title=(
            "Final Reranked Retrieval Performance"
        ),

        range_y=[
            0,
            1
        ]
    )


    fig.update_traces(

        texttemplate=
            "%{text:.4f}",

        textposition=
            "outside"
    )


    fig.update_layout(
        height=450
    )


    return fig


def grouped_plot(
    dataframe,
    group_column,
    title
):

    plot_df = dataframe.melt(

        id_vars=[
            group_column,
            "Questions"
        ],

        value_vars=[
            "MRR",
            "Hit",
            "Evidence_Recall",
            "Full_Coverage",
            "nDCG"
        ],

        var_name=
            "Metric",

        value_name=
            "Score"
    )


    plot_df[
        "Metric"
    ] = (

        plot_df[
            "Metric"
        ]

        .str.replace(
            "_",
            " "
        )
    )


    fig = px.bar(

        plot_df,

        x=group_column,

        y="Score",

        color="Metric",

        barmode="group",

        text="Score",

        title=title,

        range_y=[
            0,
            1
        ]
    )


    fig.update_traces(

        texttemplate=
            "%{text:.3f}",

        textposition=
            "outside"
    )


    fig.update_layout(
        height=520
    )


    return fig


def rank_plot(
    results_df
):

    ranks = pd.to_numeric(

        results_df.loc[

            results_df[
                "First Evidence Rank"
            ]
            !=
            "Not Retrieved",

            "First Evidence Rank"
        ],

        errors=
            "coerce"
    ).dropna()


    rank_df = (

        ranks

        .value_counts()

        .sort_index()

        .rename_axis(
            "Rank"
        )

        .reset_index(
            name=
                "Questions"
        )
    )


    fig = px.bar(

        rank_df,

        x="Rank",

        y="Questions",

        text="Questions",

        title=(
            "Rank of First Retrieved Evidence"
        )
    )


    fig.update_layout(
        height=420
    )


    return fig


# ============================================================
# GRADIO CALLBACK
# ============================================================

def run_evaluation(
    progress=gr.Progress()
):

    (
        overall,
        results_df,
        summary_df,
        category_df,
        difficulty_df,
        hops_df

    ) = evaluate_pipeline(
        progress=progress
    )


    runtime_minutes = (

        overall[
            "Runtime Seconds"
        ]

        /
        60
    )


    summary = f"""
# ResearchPilot — Final Retrieval Evaluation

### Pipeline

**Qwen3-Embedding-8B → Dense Top {CANDIDATE_K} → Qwen3-Reranker-4B → Final Top {FINAL_K}**

---

### Results

**Questions:** {overall['Questions']}

**MRR@{FINAL_K}:** {overall['MRR']:.4f}

**Hit@{FINAL_K}:** {overall['Hit']:.4f}

**Evidence Recall@{FINAL_K}:** {overall['Evidence Recall']:.4f}

**Full Coverage@{FINAL_K}:** {overall['Full Coverage']:.4f}

**Project nDCG@{FINAL_K}:** {overall['nDCG']:.4f}

---

**Evaluation runtime:** {runtime_minutes:.1f} minutes

**Evidence threshold:** {EVIDENCE_MATCH_THRESHOLD:.2f}

This benchmark evaluates only the retrieval and reranking layer.

Source-aware company filtering, calculator tool use, and LLM answer
generation are evaluated separately in the final answer evaluation.
"""


    return (

        summary,

        overall_plot(
            overall
        ),

        grouped_plot(
            category_df,
            "Category",
            "Performance by Question Category"
        ),

        grouped_plot(
            difficulty_df,
            "Difficulty",
            "Performance by Difficulty"
        ),

        grouped_plot(
            hops_df,
            "Hops",
            "Performance by Number of Hops"
        ),

        rank_plot(
            results_df
        ),

        summary_df.round(
            4
        ),

        category_df.round(
            4
        ),

        difficulty_df.round(
            4
        ),

        hops_df.round(
            4
        ),

        results_df,

        str(
            RESULTS_CSV_PATH
        ),

        str(
            SUMMARY_CSV_PATH
        )
    )


# ============================================================
# GRADIO DASHBOARD
# ============================================================

with gr.Blocks(
    title=(
        "ResearchPilot — "
        "Final Retrieval Evaluation"
    )
) as demo:


    gr.Markdown(
        f"""
# ResearchPilot — Evidence-Level Retrieval Evaluator

### Final retrieval architecture

**Qwen3-Embedding-8B → Top {CANDIDATE_K} → Qwen3-Reranker-4B → Top {FINAL_K}**

This is the primary retrieval benchmark for ResearchPilot.
"""
    )


    run_button = gr.Button(
        "Run Final Evaluation",
        variant="primary"
    )


    summary_output = gr.Markdown()


    gr.Markdown(
        "## Overall Metrics"
    )

    overall_chart = gr.Plot()


    gr.Markdown(
        "## Performance by Category"
    )

    category_chart = gr.Plot()


    gr.Markdown(
        "## Performance by Difficulty"
    )

    difficulty_chart = gr.Plot()


    gr.Markdown(
        "## Performance by Number of Hops"
    )

    hops_chart = gr.Plot()


    gr.Markdown(
        "## First Evidence Rank"
    )

    first_rank_chart = gr.Plot()


    gr.Markdown(
        "## Overall Summary"
    )

    summary_table = gr.Dataframe(
        interactive=False
    )


    gr.Markdown(
        "## Category Results"
    )

    category_table = gr.Dataframe(
        interactive=False
    )


    gr.Markdown(
        "## Difficulty Results"
    )

    difficulty_table = gr.Dataframe(
        interactive=False
    )


    gr.Markdown(
        "## Hop Results"
    )

    hops_table = gr.Dataframe(
        interactive=False
    )


    gr.Markdown(
        "## Per-Question Results"
    )

    results_table = gr.Dataframe(
        interactive=False
    )


    gr.Markdown(
        "## Exported Results"
    )


    with gr.Row():

        detailed_file = gr.File(
            label=(
                "Detailed Results CSV"
            )
        )

        summary_file = gr.File(
            label=(
                "Summary CSV"
            )
        )


    run_button.click(

        fn=
            run_evaluation,

        inputs=[],

        outputs=[

            summary_output,

            overall_chart,

            category_chart,

            difficulty_chart,

            hops_chart,

            first_rank_chart,

            summary_table,

            category_table,

            difficulty_table,

            hops_table,

            results_table,

            detailed_file,

            summary_file,
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    share=True
)