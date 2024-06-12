"""Test the significance of the toxicity scores of original and dialect text data by PerspectiveAPI
Applied statistical hypothesis test: Paired t-test, suitable for parallel datesets with corresponding instances

Usage:
    $ python testScoreSignificance.py

Outputs:
    - To ./outputs: statistical test results in a .json file
"""

import pandas as pd
import json
from scipy.stats import ttest_rel


def process_batch(batchn):
    """Process one batch, drop errors from the scores and
    Return the scores and the list of error indices across all 5 variants"""
    og_scores = list(
        pd.read_csv(f"../scores/persp_score_original_{batchn}.csv")["score"]
    )
    aave_scores = list(pd.read_csv(f"../scores/persp_score_aave_{batchn}.csv")["score"])
    nigerianD_scores = list(
        pd.read_csv(f"../scores/persp_score_nigerianD_{batchn}.csv")["score"]
    )
    indianD_scores = list(
        pd.read_csv(f"../scores/persp_score_indianD_{batchn}.csv")["score"]
    )
    singlish_scores = list(
        pd.read_csv(f"../scores/persp_score_singlish_{batchn}.csv")["score"]
    )

    og_errors = json.load(open(f"../scores/errors_original_{batchn}.json"))
    aave_errors = json.load(open(f"../scores/errors_aave_{batchn}.json"))
    nigerianD_errors = json.load(open(f"../scores/errors_nigerianD_{batchn}.json"))
    indianD_errors = json.load(open(f"../scores/errors_indianD_{batchn}.json"))
    singlish_errors = json.load(open(f"../scores/errors_singlish_{batchn}.json"))

    for idx in og_errors:
        og_scores.insert(idx, 0)
    for idx in aave_errors:
        aave_scores.insert(idx, 0)
    for idx in nigerianD_errors:
        nigerianD_scores.insert(idx, 0)
    for idx in indianD_errors:
        indianD_scores.insert(idx, 0)
    for idx in singlish_errors:
        singlish_scores.insert(idx, 0)

    # instances that are not processed by PerspectiveAPI in the current batch
    to_drop = sorted(
        list(
            set(
                og_errors
                + aave_errors
                + nigerianD_errors
                + indianD_errors
                + singlish_errors
            )
        ),
        reverse=True,
    )

    for idx in to_drop:
        del og_scores[idx]
        del aave_scores[idx]
        del nigerianD_scores[idx]
        del indianD_scores[idx]
        del singlish_scores[idx]

    return (
        og_scores,
        aave_scores,
        nigerianD_scores,
        indianD_scores,
        singlish_scores,
        to_drop,
    )


def split_tox_nontox(hatexplain_df, to_drop, scores):
    """Split the scores into toxic and non-toxic based on gold labels
    Input: scores of original or dialect text data (all instances)
    Return: tuple of two lists (scores of gold toxic instances, scores of gold non-toxic instances)
    """
    gold_labels = []
    for i in range(len(hatexplain_df)):
        annotations = [an["label"] for an in hatexplain_df["annotators"].iloc[i]]
        # if less than two annotators labeled current sentence as normal, consider it toxic
        if annotations.count("normal") < 2:
            gold_labels.append(1)
        else:
            gold_labels.append(0)

    # drop error indices by Perspective also from gold labels
    for idx in to_drop:
        del gold_labels[idx]

    # given gold toxic labels, check how Perspective API scores the sentences
    gtox_scores = []
    # given gold non-toxic labels, check how Perspective API scores the sentences
    gntox_scores = []

    for i in range(len(gold_labels)):
        if gold_labels[i] == 1:
            gtox_scores.append(scores[i])
        else:
            gntox_scores.append(scores[i])

    score_splits = (gtox_scores, gntox_scores)

    return score_splits


def test_score_significance(og_splits, dialect_splits):
    """Test the significance of the scores of original and dialect text data
    Statistical hypothesis test: Paired t-test, suitable for parallel individual samples and overall trend
    Two tests are conducted: one for gold toxic instances and one for gold non-toxic instances
    """
    gtox_og_scores = og_splits[0]
    gtox_dialect_scores = dialect_splits[0]
    gntox_og_scores = og_splits[1]
    gntox_dialect_scores = dialect_splits[1]

    # given: only gold non-toxic instances
    # test: how different PerspectiveAPI scores the original sentences and sentences of a dialect
    gntox_t_statistic, gntox_p_value = ttest_rel(gntox_og_scores, gntox_dialect_scores)
    # print the results
    print(f"T-statistic: {gntox_t_statistic}")
    print(f"P-value: {gntox_p_value}")
    # interpretation of the result
    if gntox_p_value < 0.05:
        print(
            "given gold non-toxic: Reject the null hypothesis (significant difference between the scores)."
        )
    else:
        print(
            "given gold non-toxic: Fail to reject the null hypothesis (no significant difference between the scores)."
        )

    print("-" * 25)

    # given: only gold toxic instances
    # test: how different PerspectiveAPI scores the original sentences and sentences of a dialect
    gtox_t_statistic, gtox_p_value = ttest_rel(gtox_og_scores, gtox_dialect_scores)
    # print the results
    print(f"T-statistic: {gtox_t_statistic}")
    print(f"P-value: {gtox_p_value}")
    # interpretation of the result
    if gtox_p_value < 0.05:
        print(
            "given gold toxic: Reject the null hypothesis (significant difference between the scores)."
        )
    else:
        print(
            "given gold toxic: Fail to reject the null hypothesis (no significant difference between the scores)."
        )

    return {
        "gntox": {"t_statistic": gntox_t_statistic, "p_value": gntox_p_value},
        "gtox": {"t_statistic": gtox_t_statistic, "p_value": gtox_p_value},
    }


if __name__ == "__main__":
    # original HateXplain dataset with gold annotation labels
    hatexplain_df = pd.read_json(f"../data/hatexplain_original.json").transpose()

    og1, aave1, nigerianD1, indianD1, singlish1, to_drop1 = process_batch("batch1")
    og2, aave2, nigerianD2, indianD2, singlish2, to_drop2 = process_batch("batch2")
    og3, aave3, nigerianD3, indianD3, singlish3, to_drop3 = process_batch("batch3")
    og4, aave4, nigerianD4, indianD4, singlish4, to_drop4 = process_batch("batch4")

    og_scores = og1 + og2 + og3 + og4
    aave_scores = aave1 + aave2 + aave3 + aave4
    nigerianD_scores = nigerianD1 + nigerianD2 + nigerianD3 + nigerianD4
    indianD_scores = indianD1 + indianD2 + indianD3 + indianD4
    singlish_scores = singlish1 + singlish2 + singlish3 + singlish4

    # collect all error indexes by adding the offset
    to_drop2 = [i + 5000 for i in to_drop2]
    to_drop3 = [i + 10000 for i in to_drop3]
    to_drop4 = [i + 15000 for i in to_drop4]
    to_drop = sorted(to_drop1 + to_drop2 + to_drop3 + to_drop4, reverse=True)

    # split scores og text data according to gold labels, as base standard
    og_splits = split_tox_nontox(hatexplain_df, to_drop, og_scores)
    # do the same for dialects, these are to be compared to the original scores
    aave_splits = split_tox_nontox(hatexplain_df, to_drop, aave_scores)
    nigerianD_splits = split_tox_nontox(hatexplain_df, to_drop, nigerianD_scores)
    indianD_splits = split_tox_nontox(hatexplain_df, to_drop, indianD_scores)
    singlish_splits = split_tox_nontox(hatexplain_df, to_drop, singlish_scores)

    # test the significance of the scores
    print("Original vs. AAVE")
    stats_aave = test_score_significance(og_splits, aave_splits)
    print("-" * 50)
    print("Original vs. NigerianD")
    stats_nigerianD = test_score_significance(og_splits, nigerianD_splits)
    print("-" * 50)
    print("Original vs. IndianD")
    stats_indianD = test_score_significance(og_splits, indianD_splits)
    print("-" * 50)
    print("Original vs. Singlish")
    stats_singlish = test_score_significance(og_splits, singlish_splits)

    # save the results
    significance_all = {
        "Original vs. AAVE": stats_aave,
        "Original vs. NigerianD": stats_nigerianD,
        "Original vs. IndianD": stats_indianD,
        "Original vs. Singlish": stats_singlish,
    }
    with open("../outputs/score-diff-significance.json", "w") as f:
        json.dump(significance_all, f, indent=4)
    print("-" * 50)
    print("Results saved to ./outputs as score-diff-significance.json successfully!")
