#!/usr/bin/env python3
"""
generate_analysis_csvs.py
=========================

RAG Dissertation — Results Analysis

Uses:
    1. baseline_100_qwen.jsonl
    2. refinement_100.jsonl

The refinement JSONL already contains the conflicting/injected
retrieval evidence and the initial evaluation, so a separate
conflicting_100_qwen.jsonl is NOT required.

Pipeline represented by refinement_100.jsonl:

    Original retrieved evidence
            ↓
    Conflict injection
            ↓
    Modified retrieved evidence
            ↓
    Initial answer
            ↓
    Faithfulness + safety evaluation
            ↓
    Decision Engine
            ↓
    Answer refinement (where applicable)
            ↓
    Post-refinement evaluation


OUTPUT FILES
------------
summary_stats.csv
question_level.csv
claim_level.csv
powerbi_flat.csv
refinement_decisions.csv
"""

import argparse
import csv
import math
import sys
import json
from pathlib import Path


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    here = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Generate dissertation analysis CSVs from baseline and refinement JSONL files."
    )

    parser.add_argument(
        "--baseline",
        default=str(here / "baseline_100_qwen.jsonl"),
        help="Path to baseline JSONL."
    )

    parser.add_argument(
        "--refinement",
        default=str(here / "refinement_100.jsonl"),
        help="Path to refinement JSONL. This contains the conflicting experiment."
    )

    parser.add_argument(
        "--output-dir",
        default=str(here),
        help="Directory for generated CSV files."
    )

    return parser.parse_args()


# ============================================================================
# JSONL LOADER
# ============================================================================

def load_jsonl(path: str) -> list:

    records = []

    with open(path, "r", encoding="utf-8") as fh:

        for lineno, line in enumerate(fh, 1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as exc:

                print(
                    f"WARNING: {path}, line {lineno}: "
                    f"JSON error: {exc}"
                )

    return records


# ============================================================================
# HELPERS
# ============================================================================

def safe(value, default=0):

    return value if value is not None else default


def pct(numerator, denominator):

    if not denominator:
        return None

    return round(100.0 * numerator / denominator, 2)


def mean(values):

    values = [
        x for x in values
        if x is not None
    ]

    if not values:
        return None

    return round(sum(values) / len(values), 4)


def stdev(values):

    values = [
        x for x in values
        if x is not None
    ]

    if len(values) < 2:
        return None

    m = sum(values) / len(values)

    return round(
        math.sqrt(
            sum((x - m) ** 2 for x in values)
            / (len(values) - 1)
        ),
        4
    )


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def get_claim_evaluations(
    record: dict,
    eval_key: str = "evaluation"
) -> list:

    evaluation = record.get(eval_key) or {}

    return evaluation.get(
        "claim_evaluations",
        []
    )


def get_summary(
    record: dict,
    eval_key: str = "evaluation"
) -> dict:

    evaluation = record.get(eval_key) or {}

    return evaluation.get(
        "summary",
        {}
    )


def faithfulness_verdicts(
    record: dict,
    eval_key: str = "evaluation"
) -> list:

    return [
        ce.get("faithfulness", "")
        for ce in get_claim_evaluations(
            record,
            eval_key
        )
    ]


def safety_verdicts(
    record: dict,
    eval_key: str = "evaluation"
) -> list:

    return [
        ce.get("safety_verdict") or "NONE"
        for ce in get_claim_evaluations(
            record,
            eval_key
        )
    ]


def get_resolution_action(record: dict) -> str:

    resolution = record.get(
        "answer_refinement_resolution"
    )

    if resolution is None:
        return "NO_REFINEMENT_NEEDED"

    if isinstance(resolution, dict):
        return resolution.get(
            "action",
            ""
        )

    return str(resolution)


# ============================================================================
# CSV WRITER
# ============================================================================

def write_csv(
    path: Path,
    rows: list,
    fieldnames=None
):

    if not rows:

        print(
            f"WARNING: No rows to write for {path.name}"
        )

        return

    if fieldnames is None:
        fieldnames = list(
            rows[0].keys()
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"✓ {path.name:<35} "
        f"({len(rows):>5} rows)"
    )


# ============================================================================
# QUESTION-LEVEL TABLE
# ============================================================================

def build_question_level(
    baseline,
    refinement
):

    rows = []

    for i, rf in enumerate(refinement):

        qid = str(
            rf.get(
                "question_id",
                i
            )
        )

        # ------------------------------------------------------------
        # Match baseline
        # ------------------------------------------------------------

        baseline_record = None

        for b in baseline:

            if str(
                b.get(
                    "question_id",
                    ""
                )
            ) == qid:

                baseline_record = b
                break

        if baseline_record is None:
            baseline_record = {}

        # ------------------------------------------------------------
        # Baseline evaluation
        # ------------------------------------------------------------

        b_summary = get_summary(
            baseline_record
        )

        b_faiths = faithfulness_verdicts(
            baseline_record
        )

        b_n = len(b_faiths)

        # ------------------------------------------------------------
        # Refinement initial = CONFLICTING condition
        # ------------------------------------------------------------

        rf_summary = get_summary(
            rf
        )

        rf_faiths = faithfulness_verdicts(
            rf
        )

        rf_n = len(rf_faiths)

        # ------------------------------------------------------------
        # Decision Engine
        # ------------------------------------------------------------

        decision = rf.get(
            "evidence_refinement_decision",
            {}
        ) or {}

        # ------------------------------------------------------------
        # Answer Refinement
        # ------------------------------------------------------------

        ar_eval = rf.get(
            "answer_refinement_evaluation"
        )

        ar_summary = (
            ar_eval.get("summary", {})
            if ar_eval
            else {}
        )

        ar_faiths = (
            faithfulness_verdicts(
                rf,
                "answer_refinement_evaluation"
            )
            if ar_eval
            else []
        )

        # ------------------------------------------------------------
        # Row
        # ------------------------------------------------------------

        rows.append({

            "question_id": qid,

            "question_text": (
                rf.get(
                    "question",
                    ""
                )[:120]
            ),

            # ========================================================
            # BASELINE
            # ========================================================

            "baseline_is_faithful":
                int(
                    safe(
                        b_summary.get(
                            "is_faithful"
                        ),
                        False
                    )
                ),

            "baseline_is_safe":
                int(
                    safe(
                        b_summary.get(
                            "is_safe"
                        ),
                        True
                    )
                ),

            "baseline_num_claims":
                safe(
                    b_summary.get(
                        "num_claims"
                    ),
                    0
                ),

            "baseline_num_unfaithful":
                safe(
                    b_summary.get(
                        "num_unfaithful_claims"
                    ),
                    0
                ),

            "baseline_num_unsafe":
                safe(
                    b_summary.get(
                        "num_unsafe_claims"
                    ),
                    0
                ),

            "baseline_pct_supported":
                pct(
                    b_faiths.count(
                        "SUPPORTED"
                    ),
                    b_n
                ),

            "baseline_pct_unsupported":
                pct(
                    b_faiths.count(
                        "UNSUPPORTED"
                    ),
                    b_n
                ),

            "baseline_pct_contradicted":
                pct(
                    b_faiths.count(
                        "CONTRADICTED"
                    ),
                    b_n
                ),

            "baseline_generation_time_s":
                baseline_record.get(
                    "generation_time"
                ),

            "baseline_answer":
                baseline_record.get(
                    "answer",
                    ""
                ),

            # ========================================================
            # CONFLICTING / INITIAL
            #
            # This comes directly from refinement_100.jsonl.
            # The answer/evaluation correspond to the answer generated
            # using modified_retrieved_chunks.
            # ========================================================

            "conflicting_is_faithful":
                int(
                    safe(
                        rf_summary.get(
                            "is_faithful"
                        ),
                        False
                    )
                ),

            "conflicting_is_safe":
                int(
                    safe(
                        rf_summary.get(
                            "is_safe"
                        ),
                        True
                    )
                ),

            "conflicting_num_claims":
                safe(
                    rf_summary.get(
                        "num_claims"
                    ),
                    0
                ),

            "conflicting_num_unfaithful":
                safe(
                    rf_summary.get(
                        "num_unfaithful_claims"
                    ),
                    0
                ),

            "conflicting_num_unsafe":
                safe(
                    rf_summary.get(
                        "num_unsafe_claims"
                    ),
                    0
                ),

            "conflicting_pct_supported":
                pct(
                    rf_faiths.count(
                        "SUPPORTED"
                    ),
                    rf_n
                ),

            "conflicting_pct_unsupported":
                pct(
                    rf_faiths.count(
                        "UNSUPPORTED"
                    ),
                    rf_n
                ),

            "conflicting_pct_contradicted":
                pct(
                    rf_faiths.count(
                        "CONTRADICTED"
                    ),
                    rf_n
                ),

            "conflicting_generation_time_s":
                rf.get(
                    "generation_time"
                ),

            "conflicting_answer":
                rf.get(
                    "answer",
                    ""
                ),

            # ========================================================
            # CONFLICT INJECTION CHECK
            # ========================================================

            "original_chunks_count":
                len(
                    rf.get(
                        "original_retrieved_chunks",
                        []
                    )
                ),

            "modified_chunks_count":
                len(
                    rf.get(
                        "modified_retrieved_chunks",
                        []
                    )
                ),

            "evidence_modified":
                int(
                    rf.get(
                        "original_retrieved_chunks"
                    )
                    !=
                    rf.get(
                        "modified_retrieved_chunks"
                    )
                ),

            # ========================================================
            # DECISION ENGINE
            # ========================================================

            "decision_action":
                decision.get(
                    "action",
                    ""
                ),

            "decision_reason":
                (
                    decision.get(
                        "reason",
                        ""
                    ) or ""
                )[:500],

            "decision_trigger_count":
                len(
                    decision.get(
                        "triggering_claims",
                        []
                    )
                ),

            "contradicting_chunk_count":
                len(
                    decision.get(
                        "contradicting_chunk_indices",
                        []
                    )
                ),

            "supporting_chunk_count":
                len(
                    decision.get(
                        "supporting_chunk_indices",
                        []
                    )
                ),

            # ========================================================
            # ANSWER REFINEMENT
            # ========================================================

            "answer_refinement_ran":
                int(
                    ar_eval is not None
                ),

            "after_ar_is_faithful":
                (
                    int(
                        safe(
                            ar_summary.get(
                                "is_faithful"
                            ),
                            False
                        )
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_is_safe":
                (
                    int(
                        safe(
                            ar_summary.get(
                                "is_safe"
                            ),
                            True
                        )
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_num_claims":
                (
                    safe(
                        ar_summary.get(
                            "num_claims"
                        ),
                        0
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_num_unfaithful":
                (
                    safe(
                        ar_summary.get(
                            "num_unfaithful_claims"
                        ),
                        0
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_num_unsafe":
                (
                    safe(
                        ar_summary.get(
                            "num_unsafe_claims"
                        ),
                        0
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_pct_supported":
                (
                    pct(
                        ar_faiths.count(
                            "SUPPORTED"
                        ),
                        len(ar_faiths)
                    )
                    if ar_eval
                    else ""
                ),

            "answer_refinement_resolution":
                get_resolution_action(
                    rf
                ),

            "refined_answer":
                rf.get(
                    "answer_refinement_answer",
                    ""
                ),

        })

    return rows


# ============================================================================
# CLAIM LEVEL
# ============================================================================

def build_claim_level(
    baseline,
    refinement
):

    rows = []

    def add_claims(
        records,
        condition,
        eval_key="evaluation"
    ):

        for i, record in enumerate(records):

            qid = str(
                record.get(
                    "question_id",
                    i
                )
            )

            claims = get_claim_evaluations(
                record,
                eval_key
            )

            for j, ce in enumerate(claims):

                claim_obj = ce.get(
                    "claim",
                    {}
                )

                if isinstance(
                    claim_obj,
                    dict
                ):

                    claim_text = claim_obj.get(
                        "text",
                        ""
                    )

                    claim_id = claim_obj.get(
                        "claim_id",
                        ""
                    )

                else:

                    claim_text = str(
                        claim_obj
                    )

                    claim_id = ""

                attributions = ce.get(
                    "attributions",
                    []
                )

                n_supports = sum(
                    1
                    for a in attributions
                    if a.get("label") == "SUPPORTS"
                )

                n_contradicts = sum(
                    1
                    for a in attributions
                    if a.get("label") == "CONTRADICTS"
                )

                n_neutral = sum(
                    1
                    for a in attributions
                    if a.get("label") == "NEUTRAL"
                )

                total = (
                    n_supports
                    + n_contradicts
                    + n_neutral
                )

                rows.append({

                    "condition":
                        condition,

                    "question_id":
                        qid,

                    "claim_index":
                        j,

                    "claim_id":
                        claim_id,

                    "claim_text":
                        claim_text,

                    "faithfulness":
                        ce.get(
                            "faithfulness",
                            ""
                        ),

                    "safety_verdict":
                        ce.get(
                            "safety_verdict"
                        ) or "NONE",

                    "severity":
                        ce.get(
                            "severity"
                        ) or "NONE",

                    "n_attributions":
                        total,

                    "n_supports":
                        n_supports,

                    "n_contradicts":
                        n_contradicts,

                    "n_neutral":
                        n_neutral,

                    "pct_supports":
                        pct(
                            n_supports,
                            total
                        ),

                    "pct_contradicts":
                        pct(
                            n_contradicts,
                            total
                        ),

                    "pct_neutral":
                        pct(
                            n_neutral,
                            total
                        ),

                })

    # ------------------------------------------------------------
    # Baseline claims
    # ------------------------------------------------------------

    add_claims(
        baseline,
        "baseline"
    )

    # ------------------------------------------------------------
    # Conflicting claims
    #
    # These are the INITIAL evaluations from refinement_100.jsonl.
    # ------------------------------------------------------------

    add_claims(
        refinement,
        "conflicting_initial"
    )

    # ------------------------------------------------------------
    # After Answer Refinement
    # ------------------------------------------------------------

    for i, record in enumerate(
        refinement
    ):

        if record.get(
            "answer_refinement_evaluation"
        ):

            add_claims(
                [record],
                "refinement_after_ar",
                eval_key="answer_refinement_evaluation"
            )

    return rows


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def build_summary_stats(
    baseline,
    refinement
):

    def summarise(
        records,
        condition,
        eval_key="evaluation"
    ):

        n = len(records)

        if n == 0:

            return {
                "condition": condition,
                "n_questions": 0
            }

        faithful_answers = [
            safe(
                get_summary(
                    r,
                    eval_key
                ).get(
                    "is_faithful"
                ),
                False
            )
            for r in records
        ]

        safe_answers = [
            safe(
                get_summary(
                    r,
                    eval_key
                ).get(
                    "is_safe"
                ),
                True
            )
            for r in records
        ]

        claims_per_question = [
            safe(
                get_summary(
                    r,
                    eval_key
                ).get(
                    "num_claims"
                ),
                0
            )
            for r in records
        ]

        unfaithful_per_question = [
            safe(
                get_summary(
                    r,
                    eval_key
                ).get(
                    "num_unfaithful_claims"
                ),
                0
            )
            for r in records
        ]

        unsafe_per_question = [
            safe(
                get_summary(
                    r,
                    eval_key
                ).get(
                    "num_unsafe_claims"
                ),
                0
            )
            for r in records
        ]

        generation_times = [
            r.get(
                "generation_time"
            )
            for r in records
            if r.get(
                "generation_time"
            ) is not None
        ]

        # ------------------------------------------------------------
        # Claim-level totals
        # ------------------------------------------------------------

        all_faithfulness = []
        all_safety = []

        for record in records:

            for ce in get_claim_evaluations(
                record,
                eval_key
            ):

                all_faithfulness.append(
                    ce.get(
                        "faithfulness",
                        ""
                    )
                )

                all_safety.append(
                    ce.get(
                        "safety_verdict"
                    ) or "NONE"
                )

        total_claims = len(
            all_faithfulness
        )

        n_supported = all_faithfulness.count(
            "SUPPORTED"
        )

        n_unsupported = all_faithfulness.count(
            "UNSUPPORTED"
        )

        n_contradicted = all_faithfulness.count(
            "CONTRADICTED"
        )

        n_unsafe = all_safety.count(
            "UNSAFE"
        )

        return {

            "condition":
                condition,

            "n_questions":
                n,

            "n_faithful":
                sum(
                    faithful_answers
                ),

            "pct_faithful":
                pct(
                    sum(
                        faithful_answers
                    ),
                    n
                ),

            "n_safe":
                sum(
                    safe_answers
                ),

            "pct_safe":
                pct(
                    sum(
                        safe_answers
                    ),
                    n
                ),

            "total_claims":
                total_claims,

            "n_supported_claims":
                n_supported,

            "n_unsupported_claims":
                n_unsupported,

            "n_contradicted_claims":
                n_contradicted,

            "pct_supported_claims":
                pct(
                    n_supported,
                    total_claims
                ),

            "pct_unsupported_claims":
                pct(
                    n_unsupported,
                    total_claims
                ),

            "pct_contradicted_claims":
                pct(
                    n_contradicted,
                    total_claims
                ),

            "n_unsafe_claims":
                n_unsafe,

            "pct_unsafe_claims":
                pct(
                    n_unsafe,
                    total_claims
                ),

            "mean_claims_per_question":
                mean(
                    claims_per_question
                ),

            "mean_unfaithful_per_question":
                mean(
                    unfaithful_per_question
                ),

            "mean_unsafe_per_question":
                mean(
                    unsafe_per_question
                ),

            "mean_generation_time_s":
                mean(
                    generation_times
                ),

            "std_generation_time_s":
                stdev(
                    generation_times
                ),

            "min_generation_time_s":
                (
                    round(
                        min(generation_times),
                        4
                    )
                    if generation_times
                    else None
                ),

            "max_generation_time_s":
                (
                    round(
                        max(generation_times),
                        4
                    )
                    if generation_times
                    else None
                ),
        }

    # ------------------------------------------------------------
    # Three actual conditions
    # ------------------------------------------------------------

    rows = [

        summarise(
            baseline,
            "baseline"
        ),

        summarise(
            refinement,
            "conflicting_initial"
        ),

    ]

    # ------------------------------------------------------------
    # Answer refinement subset
    # ------------------------------------------------------------

    records_after_ar = [

        r
        for r in refinement
        if r.get(
            "answer_refinement_evaluation"
        )
    ]

    rows.append(

        summarise(
            records_after_ar,
            f"refinement_after_ar (n={len(records_after_ar)})",
            eval_key="answer_refinement_evaluation"
        )

    )

    # ------------------------------------------------------------
    # Baseline vs conflicting delta
    # ------------------------------------------------------------

    if len(rows) >= 2:

        baseline_row = rows[0]
        conflicting_row = rows[1]

        for key in [

            "pct_faithful",
            "pct_safe",
            "pct_supported_claims",
            "pct_unsupported_claims",
            "pct_contradicted_claims",
            "mean_unfaithful_per_question",
            "mean_unsafe_per_question",

        ]:

            b = baseline_row.get(key)
            c = conflicting_row.get(key)

            if (
                b is not None
                and c is not None
            ):

                conflicting_row[
                    f"delta_vs_baseline_{key}"
                ] = round(
                    c - b,
                    2
                )

    return rows


# ============================================================================
# POWER BI
# ============================================================================

def build_powerbi_flat(
    baseline,
    refinement
):

    rows = []

    def make_row(
        record,
        condition,
        eval_key="evaluation",
        qid_override=None
    ):

        summary = get_summary(
            record,
            eval_key
        )

        faithfulness = faithfulness_verdicts(
            record,
            eval_key
        )

        safety = safety_verdicts(
            record,
            eval_key
        )

        n = len(
            faithfulness
        )

        return {

            "question_id":
                qid_override
                or record.get(
                    "question_id",
                    ""
                ),

            "condition":
                condition,

            "failure_mode":
                record.get(
                    "failure_mode",
                    ""
                ),

            "is_faithful":
                int(
                    safe(
                        summary.get(
                            "is_faithful"
                        ),
                        False
                    )
                ),

            "is_safe":
                int(
                    safe(
                        summary.get(
                            "is_safe"
                        ),
                        True
                    )
                ),

            "num_claims":
                safe(
                    summary.get(
                        "num_claims"
                    ),
                    0
                ),

            "num_unfaithful":
                safe(
                    summary.get(
                        "num_unfaithful_claims"
                    ),
                    0
                ),

            "num_unsafe":
                safe(
                    summary.get(
                        "num_unsafe_claims"
                    ),
                    0
                ),

            "pct_supported":
                pct(
                    faithfulness.count(
                        "SUPPORTED"
                    ),
                    n
                ),

            "pct_unsupported":
                pct(
                    faithfulness.count(
                        "UNSUPPORTED"
                    ),
                    n
                ),

            "pct_contradicted":
                pct(
                    faithfulness.count(
                        "CONTRADICTED"
                    ),
                    n
                ),

            "n_safe_claims":
                safety.count(
                    "SAFE"
                ),

            "n_unsafe_claims":
                safety.count(
                    "UNSAFE"
                ),

            "generation_time_s":
                record.get(
                    "generation_time"
                ),

            "answer_length_words":
                len(
                    (
                        record.get(
                            "answer"
                        )
                        or ""
                    ).split()
                ),

        }

    # ------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------

    for record in baseline:

        rows.append(
            make_row(
                record,
                "Baseline"
            )
        )

    # ------------------------------------------------------------
    # Conflicting initial
    #
    # Comes from refinement JSONL.
    # ------------------------------------------------------------

    for i, record in enumerate(
        refinement
    ):

        rows.append(
            make_row(
                record,
                "Conflicting",
                qid_override=str(i)
            )
        )

        # --------------------------------------------------------
        # After Answer Refinement
        # --------------------------------------------------------

        if record.get(
            "answer_refinement_evaluation"
        ):

            rows.append(
                make_row(
                    record,
                    "Refinement-AfterAR",
                    eval_key="answer_refinement_evaluation",
                    qid_override=str(i)
                )
            )

    return rows


# ============================================================================
# DECISION ENGINE TABLE
# ============================================================================

def build_refinement_decisions(
    refinement
):

    rows = []

    for i, record in enumerate(
        refinement
    ):

        decision = record.get(
            "evidence_refinement_decision",
            {}
        ) or {}

        ar_eval = record.get(
            "answer_refinement_evaluation"
        )

        ar_summary = (
            ar_eval.get(
                "summary",
                {}
            )
            if ar_eval
            else {}
        )

        initial_summary = get_summary(
            record
        )

        rows.append({

            "question_id":
                str(i),

            "question_text":
                record.get(
                    "question",
                    ""
                )[:200],

            "failure_mode":
                record.get(
                    "failure_mode",
                    ""
                ),

            # ----------------------------------------------------
            # Evidence modification
            # ----------------------------------------------------

            "evidence_modified":
                int(
                    record.get(
                        "original_retrieved_chunks"
                    )
                    !=
                    record.get(
                        "modified_retrieved_chunks"
                    )
                ),

            "original_chunk_count":
                len(
                    record.get(
                        "original_retrieved_chunks",
                        []
                    )
                ),

            "modified_chunk_count":
                len(
                    record.get(
                        "modified_retrieved_chunks",
                        []
                    )
                ),

            # ----------------------------------------------------
            # Initial evaluation
            # ----------------------------------------------------

            "initial_is_faithful":
                int(
                    safe(
                        initial_summary.get(
                            "is_faithful"
                        ),
                        False
                    )
                ),

            "initial_is_safe":
                int(
                    safe(
                        initial_summary.get(
                            "is_safe"
                        ),
                        True
                    )
                ),

            "initial_num_claims":
                safe(
                    initial_summary.get(
                        "num_claims"
                    ),
                    0
                ),

            "initial_num_unfaithful":
                safe(
                    initial_summary.get(
                        "num_unfaithful_claims"
                    ),
                    0
                ),

            "initial_num_unsafe":
                safe(
                    initial_summary.get(
                        "num_unsafe_claims"
                    ),
                    0
                ),

            # ----------------------------------------------------
            # Decision
            # ----------------------------------------------------

            "decision_action":
                decision.get(
                    "action",
                    ""
                ),

            "decision_reason":
                (
                    decision.get(
                        "reason",
                        ""
                    )
                    or ""
                ),

            "n_triggering_claims":
                len(
                    decision.get(
                        "triggering_claims",
                        []
                    )
                ),

            "n_contradicting_chunks":
                len(
                    decision.get(
                        "contradicting_chunk_indices",
                        []
                    )
                ),

            "n_supporting_chunks":
                len(
                    decision.get(
                        "supporting_chunk_indices",
                        []
                    )
                ),

            # ----------------------------------------------------
            # Answer refinement
            # ----------------------------------------------------

            "answer_refinement_ran":
                int(
                    ar_eval is not None
                ),

            "after_ar_is_faithful":
                (
                    int(
                        safe(
                            ar_summary.get(
                                "is_faithful"
                            ),
                            False
                        )
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_is_safe":
                (
                    int(
                        safe(
                            ar_summary.get(
                                "is_safe"
                            ),
                            True
                        )
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_num_claims":
                (
                    safe(
                        ar_summary.get(
                            "num_claims"
                        ),
                        0
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_num_unfaithful":
                (
                    safe(
                        ar_summary.get(
                            "num_unfaithful_claims"
                        ),
                        0
                    )
                    if ar_eval
                    else ""
                ),

            "after_ar_num_unsafe":
                (
                    safe(
                        ar_summary.get(
                            "num_unsafe_claims"
                        ),
                        0
                    )
                    if ar_eval
                    else ""
                ),

            "resolution_action":
                get_resolution_action(
                    record
                ),

            "refined_answer":
                record.get(
                    "answer_refinement_answer",
                    ""
                ),

        })

    return rows


# ============================================================================
# CONSOLE SUMMARY
# ============================================================================

def print_summary(
    summary_rows
):

    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for row in summary_rows:

        print()

        print(
            f"[{row['condition']}] "
            f"N = {row['n_questions']}"
        )

        print(
            f"  Faithful answers   : "
            f"{row.get('n_faithful', 0)}/"
            f"{row['n_questions']} "
            f"({row.get('pct_faithful')}%)"
        )

        print(
            f"  Safe answers       : "
            f"{row.get('n_safe', 0)}/"
            f"{row['n_questions']} "
            f"({row.get('pct_safe')}%)"
        )

        print(
            f"  Supported claims   : "
            f"{row.get('n_supported_claims', 0)}/"
            f"{row.get('total_claims', 0)} "
            f"({row.get('pct_supported_claims')}%)"
        )

        print(
            f"  Unsupported claims : "
            f"{row.get('n_unsupported_claims', 0)}/"
            f"{row.get('total_claims', 0)} "
            f"({row.get('pct_unsupported_claims')}%)"
        )

        print(
            f"  Contradicted claims: "
            f"{row.get('n_contradicted_claims', 0)}/"
            f"{row.get('total_claims', 0)} "
            f"({row.get('pct_contradicted_claims')}%)"
        )

        print(
            f"  Avg generation time: "
            f"{row.get('mean_generation_time_s')}s"
        )

    print()
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------

    required_files = [

        (
            "baseline",
            args.baseline
        ),

        (
            "refinement",
            args.refinement
        ),

    ]

    for label, path in required_files:

        if not Path(path).exists():

            print(
                f"ERROR: {label} file not found:\n"
                f"{path}",
                file=sys.stderr
            )

            sys.exit(1)

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    print()
    print("Loading JSONL files...")
    print()

    baseline = load_jsonl(
        args.baseline
    )

    refinement = load_jsonl(
        args.refinement
    )

    print(
        f"  baseline   : "
        f"{len(baseline)} records"
    )

    print(
        f"  refinement : "
        f"{len(refinement)} records"
    )

    print()

    # ------------------------------------------------------------
    # Build CSVs
    # ------------------------------------------------------------

    print(
        f"Writing CSVs to:\n"
        f"{output_dir}"
    )

    print()

    summary_rows = build_summary_stats(
        baseline,
        refinement
    )

    write_csv(
        output_dir / "summary_stats.csv",
        summary_rows
    )

    question_rows = build_question_level(
        baseline,
        refinement
    )

    write_csv(
        output_dir / "question_level.csv",
        question_rows
    )

    claim_rows = build_claim_level(
        baseline,
        refinement
    )

    write_csv(
        output_dir / "claim_level.csv",
        claim_rows
    )

    powerbi_rows = build_powerbi_flat(
        baseline,
        refinement
    )

    write_csv(
        output_dir / "powerbi_flat.csv",
        powerbi_rows
    )

    decision_rows = build_refinement_decisions(
        refinement
    )

    write_csv(
        output_dir / "refinement_decisions.csv",
        decision_rows
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print_summary(
        summary_rows
    )

    print()
    print("DONE.")
    print()
    print("Generated:")
    print("  summary_stats.csv")
    print("  question_level.csv")
    print("  claim_level.csv")
    print("  powerbi_flat.csv")
    print("  refinement_decisions.csv")


if __name__ == "__main__":
    main()