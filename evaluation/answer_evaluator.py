# ============================================================
# ResearchPilot — FINAL ANSWER QUALITY EVALUATION
#
# Assumes this already exists:
#
# question_answer(question)
#
# which returns:
# (
#     answer,
#     reranked_contexts,
#     reranked_metadatas
# )
# ============================================================
# IMPORTS
# ============================================================

import json
from pathlib import Path
from typing import Literal
from IPython.display import display

import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)
from retrieval import question_answer

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GOLDEN_DATASET_PATH = (BASE_DIR/"golden_dataset.jsonl")

OUTPUT_PATH = (BASE_DIR/"final_answer_evaluation_v2.csv")

JUDGE_MODEL = "gpt-5.4-mini"


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

with open(
    GOLDEN_DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    golden_dataset = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print(
    f"Loaded {len(golden_dataset)} questions."
)


# ============================================================
# STRUCTURED JUDGE OUTPUT
# ============================================================

class AnswerEvaluation(BaseModel):

    correctness: Literal[
        "pass",
        "partial",
        "fail"
    ]

    groundedness: Literal[
        "pass",
        "partial",
        "fail"
    ]

    completeness: Literal[
        "pass",
        "partial",
        "fail"
    ]

    reason: str


# ============================================================
# JUDGE MODEL
# ============================================================

judge_llm = ChatOpenAI(
    model=JUDGE_MODEL
)


judge = (
    judge_llm
    .with_structured_output(
        AnswerEvaluation
    )
)


# ============================================================
# JUDGE INSTRUCTIONS
# ============================================================

JUDGE_SYSTEM_PROMPT = """
You are evaluating the output of a RAG system.

Use ONLY the information supplied in this evaluation prompt.
Do not use outside knowledge.

Evaluate three dimensions:

1. CORRECTNESS

Compare the generated answer with the expected answer and
golden evidence claims.

PASS:
The answer is factually correct and agrees with the expected answer.

PARTIAL:
The answer is broadly correct but contains a minor factual error,
imprecision, or only part of the correct result.

FAIL:
The answer is materially incorrect or contradicts the expected answer.


2. GROUNDEDNESS

Check whether factual claims in the generated answer are supported by
the RETRIEVED CONTEXT that was actually supplied to the answering model.

PASS:
All meaningful factual claims are supported by the retrieved context.

PARTIAL:
The central answer is supported but there is a minor unsupported claim
or extrapolation.

FAIL:
An important claim is unsupported by or contradicts the retrieved
context.


3. COMPLETENESS

Compare the generated answer with the expected answer and golden
evidence claims.

PASS:
All important information required to answer the question is present.

PARTIAL:
The answer contains the main answer but omits one or more meaningful
details.

FAIL:
A major part of the requested answer is missing.


Additional rules:

- Ignore differences in wording and formatting.
- Accept equivalent units and reasonable rounding.
- Accept mathematically equivalent calculations.
- Do not penalize concise answers merely for being concise.
- Do not reward information that is correct but irrelevant.
- A refusal or "I don't know" is incorrect if the supplied golden
  information clearly contains an answer.
- Judge the answer, not the writing style.

Keep the reason concise.
"""


# ============================================================
# SCORE CONVERSION
#
# pass    = 1.0
# partial = 0.5
# fail    = 0.0
# ============================================================

SCORE_MAP = {
    "pass": 1.0,
    "partial": 0.5,
    "fail": 0.0
}


# ============================================================
# FORMAT GOLDEN CLAIMS
# ============================================================

def format_golden_claims(item):

    claims = []

    for evidence in item.get(
        "evidence",
        []
    ):

        claim = evidence.get(
            "claim",
            ""
        )

        if claim:
            claims.append(claim)

    return "\n".join(
        f"- {claim}"
        for claim in claims
    )


# ============================================================
# FORMAT RETRIEVED CONTEXT
# ============================================================

def format_retrieved_context(
    contexts,
    metadatas
):

    parts = []

    for i, (
        context,
        metadata
    ) in enumerate(

        zip(
            contexts,
            metadatas
        ),

        start=1
    ):

        source = metadata.get(
            "source",
            "Unknown"
        )

        parts.append(
            f"""
PASSAGE {i}
Source: {source}

{context}
"""
        )

    return "\n\n".join(parts)


# ============================================================
# JUDGE ONE ANSWER
# ============================================================

def judge_answer(
    item,
    generated_answer,
    contexts,
    metadatas
):

    golden_claims = (
        format_golden_claims(item)
    )

    retrieved_context = (
        format_retrieved_context(
            contexts,
            metadatas
        )
    )


    evaluation_prompt = f"""
QUESTION:
{item["question"]}


EXPECTED ANSWER:
{item.get("expected_answer", "")}


GOLDEN EVIDENCE CLAIMS:
{golden_claims}


GENERATED ANSWER:
{generated_answer}


RETRIEVED CONTEXT USED BY THE SYSTEM:
{retrieved_context}
"""


    result = judge.invoke(
        [
            SystemMessage(
                content=JUDGE_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=evaluation_prompt
            )
        ]
    )

    return result


# ============================================================
# RESUME SUPPORT
#
# Saves after EVERY question.
#
# If Kaggle crashes after question 60,
# rerunning this cell resumes at 61 rather
# than wasting all completed work.
# ============================================================

if Path(OUTPUT_PATH).exists():

    previous_results = pd.read_csv(
        OUTPUT_PATH
    )

    completed_ids = set(
        previous_results["ID"]
        .astype(str)
        .tolist()
    )

    results = (
        previous_results
        .to_dict("records")
    )

    print(
        f"Resuming evaluation. "
        f"{len(completed_ids)} questions "
        f"already completed."
    )

else:

    completed_ids = set()
    results = []


# ============================================================
# RUN FINAL END-TO-END EVALUATION
# ============================================================

for item in tqdm(
    golden_dataset,
    desc="Evaluating final RAG",
    unit="question"
):

    question_id = str(
        item["id"]
    )


    # --------------------------------------------------------
    # Skip questions already saved
    # --------------------------------------------------------

    if question_id in completed_ids:
        continue


    question = item[
        "question"
    ]


    # ========================================================
    # ACTUAL PRODUCTION PIPELINE
    #
    # Retrieval
    # → reranking
    # → optional calculator
    # → final answer
    # ========================================================

    answer, contexts, metadatas = (
        question_answer(
            question
        )
    )


    # ========================================================
    # LLM JUDGE
    # ========================================================

    evaluation = judge_answer(
        item,
        answer,
        contexts,
        metadatas
    )


    # ========================================================
    # NUMERIC SCORES
    # ========================================================

    correctness_score = SCORE_MAP[
        evaluation.correctness
    ]

    groundedness_score = SCORE_MAP[
        evaluation.groundedness
    ]

    completeness_score = SCORE_MAP[
        evaluation.completeness
    ]


    overall_score = (

        correctness_score
        +
        groundedness_score
        +
        completeness_score

    ) / 3


    # ========================================================
    # SAVE RESULT
    # ========================================================

    row = {

        "ID":
            question_id,

        "Company":
            item.get(
                "company",
                ""
            ),

        "Category":
            item.get(
                "category",
                ""
            ),

        "Difficulty":
            item.get(
                "difficulty",
                ""
            ),

        "Hops":
            item.get(
                "hops",
                1
            ),

        "Question":
            question,

        "Expected Answer":
            item.get(
                "expected_answer",
                ""
            ),

        "Generated Answer":
            answer,

        "Correctness":
            evaluation.correctness,

        "Correctness Score":
            correctness_score,

        "Groundedness":
            evaluation.groundedness,

        "Groundedness Score":
            groundedness_score,

        "Completeness":
            evaluation.completeness,

        "Completeness Score":
            completeness_score,

        "Overall Score":
            overall_score,

        "Judge Reason":
            evaluation.reason,
    }


    results.append(row)


    # --------------------------------------------------------
    # CHECKPOINT AFTER EVERY QUESTION
    # --------------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        OUTPUT_PATH,
        index=False
    )


# ============================================================
# FINAL DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# OVERALL RESULTS
# ============================================================

correctness = results_df[
    "Correctness Score"
].mean()

groundedness = results_df[
    "Groundedness Score"
].mean()

completeness = results_df[
    "Completeness Score"
].mean()

overall = results_df[
    "Overall Score"
].mean()


# Strict all-pass:
strict_pass_rate = (

    (
        (results_df["Correctness"] == "pass")
        &
        (results_df["Groundedness"] == "pass")
        &
        (results_df["Completeness"] == "pass")
    )
    .mean()
)


print("\n" + "=" * 60)

print("FINAL ANSWER EVALUATION")

print("=" * 60)

print(
    f"Questions:          "
    f"{len(results_df)}"
)

print(
    f"Correctness:        "
    f"{correctness:.4f}"
)

print(
    f"Groundedness:       "
    f"{groundedness:.4f}"
)

print(
    f"Completeness:       "
    f"{completeness:.4f}"
)

print(
    f"Overall Score:      "
    f"{overall:.4f}"
)

print(
    f"Strict Pass Rate:   "
    f"{strict_pass_rate:.4f}"
)

print("=" * 60)


# ============================================================
# OVERALL CHART
# ============================================================

overall_df = pd.DataFrame(
    {
        "Metric": [
            "Correctness",
            "Groundedness",
            "Completeness",
            "Overall"
        ],

        "Score": [
            correctness,
            groundedness,
            completeness,
            overall
        ]
    }
)


fig = px.bar(
    overall_df,
    x="Metric",
    y="Score",
    text="Score",
    title="ResearchPilot — Final Answer Quality",
    range_y=[0, 1]
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

fig.show()


# ============================================================
# CATEGORY BREAKDOWN
# ============================================================

category_df = (

    results_df

    .groupby("Category")

    .agg(

        Questions=(
            "ID",
            "count"
        ),

        Correctness=(
            "Correctness Score",
            "mean"
        ),

        Groundedness=(
            "Groundedness Score",
            "mean"
        ),

        Completeness=(
            "Completeness Score",
            "mean"
        ),

        Overall=(
            "Overall Score",
            "mean"
        )
    )

    .reset_index()
)


print("\nPERFORMANCE BY CATEGORY")

display(
    category_df.round(4)
)


# ============================================================
# DIFFICULTY BREAKDOWN
# ============================================================

difficulty_df = (

    results_df

    .groupby("Difficulty")

    .agg(

        Questions=(
            "ID",
            "count"
        ),

        Correctness=(
            "Correctness Score",
            "mean"
        ),

        Groundedness=(
            "Groundedness Score",
            "mean"
        ),

        Completeness=(
            "Completeness Score",
            "mean"
        ),

        Overall=(
            "Overall Score",
            "mean"
        )
    )

    .reset_index()
)


print("\nPERFORMANCE BY DIFFICULTY")

display(
    difficulty_df.round(4)
)


# ============================================================
# SHOW FAILURES FIRST
#
# These are the only questions you should manually inspect.
# ============================================================

failures = (

    results_df[
        results_df[
            "Overall Score"
        ] < 1.0
    ]

    .sort_values(
        "Overall Score"
    )
)


print(
    f"\nQuestions needing review: "
    f"{len(failures)}"
)


display(
    failures[
        [
            "ID",
            "Category",
            "Difficulty",
            "Question",
            "Expected Answer",
            "Generated Answer",
            "Correctness",
            "Groundedness",
            "Completeness",
            "Judge Reason"
        ]
    ]
)


print(
    f"\n✅ Full evaluation saved to:\n"
    f"{OUTPUT_PATH}"
)