import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, binomtest, spearmanr


# ============================================================
# FILES
# ============================================================

INPUT = r"D:\rag_baseline_phase1\project\analysis\question_level.csv"

OUTPUT = r"D:\rag_baseline_phase1\project\analysis\statistical_results.csv"


# ============================================================
# HELPERS
# ============================================================

def mcnemar_exact(before, after):

    data = pd.DataFrame({
        "before": before,
        "after": after
    }).dropna()

    before = data["before"].astype(int)
    after = data["after"].astype(int)

    b = ((before == 1) & (after == 0)).sum()
    c = ((before == 0) & (after == 1)).sum()

    discordant = b + c

    if discordant == 0:
        p = 1.0
    else:
        p = binomtest(
            min(b, c),
            discordant,
            0.5,
            alternative="two-sided"
        ).pvalue

    return b, c, p


def wilcoxon_test(before, after):

    data = pd.DataFrame({
        "before": before,
        "after": after
    }).dropna()

    differences = data["after"] - data["before"]

    # Remove zero differences
    differences = differences[differences != 0]

    if len(differences) == 0:
        return 0, 1.0

    result = wilcoxon(
        data.loc[differences.index, "before"],
        data.loc[differences.index, "after"]
    )

    return result.statistic, result.pvalue


def sig(p):

    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 75)
print("STATISTICAL ANALYSIS OF RAG EXPERIMENT")
print("=" * 75)

print(f"\nQuestions: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CONDITION COUNTS
# ============================================================

print("\n" + "=" * 75)
print("DECISION ACTION COUNTS")
print("=" * 75)

print(
    df["decision_action"]
    .value_counts(dropna=False)
)


# ============================================================
# 1. BASELINE vs CONFLICT
# ============================================================

print("\n" + "=" * 75)
print("1. BASELINE vs CONFLICT")
print("=" * 75)


# ------------------------------------------------------------
# FAITHFULNESS — McNEMAR
# ------------------------------------------------------------

b, c, p = mcnemar_exact(
    df["baseline_is_faithful"],
    df["conflicting_is_faithful"]
)

baseline = df["baseline_is_faithful"].mean()
conflict = df["conflicting_is_faithful"].mean()

print("\nFAITHFULNESS")
print("-" * 50)

print(f"Baseline:       {baseline:.3f} ({baseline*100:.1f}%)")
print(f"Conflict:       {conflict:.3f} ({conflict*100:.1f}%)")
print(f"Change:         {(conflict-baseline)*100:.1f} percentage points")

print(f"Discordant 1→0: {b}")
print(f"Discordant 0→1: {c}")

print(f"McNemar p-value: {p:.6f}")
print(f"Significance:    {sig(p)}")


# ------------------------------------------------------------
# SAFETY — McNEMAR
# ------------------------------------------------------------

b, c, p = mcnemar_exact(
    df["baseline_is_safe"],
    df["conflicting_is_safe"]
)

baseline = df["baseline_is_safe"].mean()
conflict = df["conflicting_is_safe"].mean()

print("\nSAFETY")
print("-" * 50)

print(f"Baseline:       {baseline:.3f} ({baseline*100:.1f}%)")
print(f"Conflict:       {conflict:.3f} ({conflict*100:.1f}%)")
print(f"Change:         {(conflict-baseline)*100:.1f} percentage points")

print(f"McNemar p-value: {p:.6f}")
print(f"Significance:    {sig(p)}")


# ------------------------------------------------------------
# SUPPORTED CLAIMS — WILCOXON
# ------------------------------------------------------------

stat, p = wilcoxon_test(
    df["baseline_pct_supported"],
    df["conflicting_pct_supported"]
)

baseline = df["baseline_pct_supported"].mean()
conflict = df["conflicting_pct_supported"].mean()

print("\nSUPPORTED CLAIMS (%)")
print("-" * 50)

print(f"Baseline mean:  {baseline:.2f}%")
print(f"Conflict mean:  {conflict:.2f}%")
print(f"Change:         {conflict-baseline:.2f} percentage points")

print(f"Wilcoxon statistic: {stat}")
print(f"p-value:            {p:.6f}")
print(f"Significance:       {sig(p)}")


# ------------------------------------------------------------
# UNSUPPORTED CLAIMS — WILCOXON
# ------------------------------------------------------------

stat, p = wilcoxon_test(
    df["baseline_pct_unsupported"],
    df["conflicting_pct_unsupported"]
)

baseline = df["baseline_pct_unsupported"].mean()
conflict = df["conflicting_pct_unsupported"].mean()

print("\nUNSUPPORTED CLAIMS (%)")
print("-" * 50)

print(f"Baseline mean:  {baseline:.2f}%")
print(f"Conflict mean:  {conflict:.2f}%")
print(f"Change:         {conflict-baseline:.2f} percentage points")

print(f"Wilcoxon statistic: {stat}")
print(f"p-value:            {p:.6f}")
print(f"Significance:       {sig(p)}")


# ============================================================
# 2. ANSWER REFINEMENT
# ============================================================

print("\n" + "=" * 75)
print("2. ANSWER REFINEMENT")
print("=" * 75)


# Only questions where Answer Refinement actually ran

ref = df[
    df["answer_refinement_ran"] == True
].copy()

print(f"\nQuestions with Answer Refinement: {len(ref)}")


# ------------------------------------------------------------
# FAITHFULNESS BEFORE vs AFTER
# ------------------------------------------------------------

b, c, p = mcnemar_exact(
    ref["conflicting_is_faithful"],
    ref["after_ar_is_faithful"]
)

before = ref["conflicting_is_faithful"].mean()
after = ref["after_ar_is_faithful"].mean()

print("\nFAITHFULNESS")
print("-" * 50)

print(f"Before refinement: {before:.3f} ({before*100:.1f}%)")
print(f"After refinement:  {after:.3f} ({after*100:.1f}%)")

print(
    f"Change:            {(after-before)*100:.1f} "
    "percentage points"
)

print(f"McNemar p-value: {p:.6f}")
print(f"Significance:    {sig(p)}")


# ------------------------------------------------------------
# SAFETY BEFORE vs AFTER
# ------------------------------------------------------------

b, c, p = mcnemar_exact(
    ref["conflicting_is_safe"],
    ref["after_ar_is_safe"]
)

before = ref["conflicting_is_safe"].mean()
after = ref["after_ar_is_safe"].mean()

print("\nSAFETY")
print("-" * 50)

print(f"Before refinement: {before:.3f} ({before*100:.1f}%)")
print(f"After refinement:  {after:.3f} ({after*100:.1f}%)")

print(
    f"Change:            {(after-before)*100:.1f} "
    "percentage points"
)

print(f"McNemar p-value: {p:.6f}")
print(f"Significance:    {sig(p)}")


# ------------------------------------------------------------
# SUPPORTED CLAIMS BEFORE vs AFTER
# ------------------------------------------------------------

stat, p = wilcoxon_test(
    ref["conflicting_pct_supported"],
    ref["after_ar_pct_supported"]
)

before = ref["conflicting_pct_supported"].mean()
after = ref["after_ar_pct_supported"].mean()

print("\nSUPPORTED CLAIMS (%)")
print("-" * 50)

print(f"Before refinement: {before:.2f}%")
print(f"After refinement:  {after:.2f}%")

print(
    f"Change:            {after-before:.2f} "
    "percentage points"
)

print(f"Wilcoxon statistic: {stat}")
print(f"p-value:            {p:.6f}")
print(f"Significance:       {sig(p)}")


# ============================================================
# 3. REFINEMENT ACTIONS
# ============================================================

print("\n" + "=" * 75)
print("3. REFINEMENT DECISIONS")
print("=" * 75)

print("\nDecision distribution:")

decision_counts = df["decision_action"].value_counts(dropna=False)

for action, count in decision_counts.items():

    percentage = count / len(df) * 100

    print(
        f"{action}: "
        f"{count} questions "
        f"({percentage:.1f}%)"
    )


# ============================================================
# 4. REFINEMENT TRIGGERS
# ============================================================

print("\n" + "=" * 75)
print("4. REFINEMENT TRIGGERS")
print("=" * 75)

print(
    "\nTrigger count statistics:"
)

print(
    df["decision_trigger_count"]
    .describe()
)


# ============================================================
# 5. CONTRADICTING CHUNKS
# ============================================================

print("\n" + "=" * 75)
print("5. CONTRADICTING RETRIEVED CHUNKS")
print("=" * 75)

print(
    df["contradicting_chunk_count"]
    .describe()
)


# ============================================================
# 6. SUMMARY TABLE
# ============================================================

summary = []

# Baseline vs conflict faithfulness

b, c, p = mcnemar_exact(
    df["baseline_is_faithful"],
    df["conflicting_is_faithful"]
)

summary.append({
    "comparison": "Baseline vs Conflict",
    "metric": "Faithfulness",
    "test": "Exact McNemar",
    "p_value": p,
    "significance": sig(p)
})


# Baseline vs conflict safety

b, c, p = mcnemar_exact(
    df["baseline_is_safe"],
    df["conflicting_is_safe"]
)

summary.append({
    "comparison": "Baseline vs Conflict",
    "metric": "Safety",
    "test": "Exact McNemar",
    "p_value": p,
    "significance": sig(p)
})


# Supported claims

stat, p = wilcoxon_test(
    df["baseline_pct_supported"],
    df["conflicting_pct_supported"]
)

summary.append({
    "comparison": "Baseline vs Conflict",
    "metric": "Supported claims %",
    "test": "Wilcoxon signed-rank",
    "p_value": p,
    "significance": sig(p)
})


# Refinement faithfulness

b, c, p = mcnemar_exact(
    ref["conflicting_is_faithful"],
    ref["after_ar_is_faithful"]
)

summary.append({
    "comparison": "Before vs After Refinement",
    "metric": "Faithfulness",
    "test": "Exact McNemar",
    "p_value": p,
    "significance": sig(p)
})


# Refinement supported claims

stat, p = wilcoxon_test(
    ref["conflicting_pct_supported"],
    ref["after_ar_pct_supported"]
)

summary.append({
    "comparison": "Before vs After Refinement",
    "metric": "Supported claims %",
    "test": "Wilcoxon signed-rank",
    "p_value": p,
    "significance": sig(p)
})


summary_df = pd.DataFrame(summary)

print("\n" + "=" * 75)
print("STATISTICAL TEST SUMMARY")
print("=" * 75)

print(
    summary_df.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    OUTPUT,
    index=False
)

print("\n")
print("Results saved to:")
print(OUTPUT)

print("\nDONE.")