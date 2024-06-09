"""Post-process PerspectiveAPI's toxicity scores (percentile) to usable format (binary)
Check how accurate the PerspectiveAPI's scores are for the HateXplain dataset compared to the gold labels

Prequisites:
    - the original HateXplain dataset in json format
    - toxicity scores by PerspectiveAPI on the original
      and all 4 dialects in csv format (multiple batches)
    - error indices of all batches for all 4 dialects in json format

Usage:
    $ python checkPerspectiveReliability.py
"""

import pandas as pd
import json

from scipy.stats import chi2_contingency


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


def print_results(
    og_scores, aave_scores, nigerianD_scores, indianD_scores, singlish_scores
):
    """Print the number of comments tagged as toxic by PerspectiveAPI in each variant
    Threshold: if scores > 0.5, the comment is toxic"""
    og_tox = [score for score in og_scores if score > 0.5]
    aave_tox = [score for score in aave_scores if score > 0.5]
    nigeriand_tox = [score for score in nigerianD_scores if score > 0.5]
    indind_tox = [score for score in indianD_scores if score > 0.5]
    singlish_tox = [score for score in singlish_scores if score > 0.5]

    print("Original: ", len(og_tox), "(toxic) /", len(og_scores), "(total)")
    print("AAVE:     ", len(aave_tox), "(toxic) /", len(aave_scores), "(total)")
    print(
        "NigerianD:", len(nigeriand_tox), "(toxic) /", len(nigerianD_scores), "(total)"
    )
    print("IndianD:  ", len(indind_tox), "(toxic) /", len(indianD_scores), "(total)")
    print(
        "Singlish: ", len(singlish_tox), "(toxic) /", len(singlish_scores), "(total)\n"
    )

    return True


def check_perspective_credibility(hatexplain_df, og_scores, to_drop):
    """Compare gold labels with PerspectiveAPI's labels on the toxicity HateXplain dataset
    Use the Chi-square test to check the Trur/False of the null hypothesis"""
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

    # indices of instances that are gold toxic
    gtox_idx = [i for i in range(len(gold_labels)) if gold_labels[i] == 1]

    # indices of instances tagged as toxic by Perspective
    tox_idx_persp_og = []
    for i in range(len(og_scores)):
        if og_scores[i] > 0.5:
            tox_idx_persp_og.append(i)

    print("gold toxic count:       ", len(gtox_idx))
    print("Perspective toxic count:", len(tox_idx_persp_og))

    # since the data are categorical, we choose he Chi-square test
    # the null hypothesis is that the two categorical variables are independent
    persp_labels_og = [
        0 if i < 0.5 else 1 for i in og_scores
    ]  # transform the scores to binary labels

    # create a contingency table for the two categorical variables
    contingency_table = pd.crosstab(
        pd.Series(gold_labels, name="gold"),
        pd.Series(persp_labels_og, name="perspective"),
    )
    # perform the Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p}")
    # p == 0 < 0.05, reject the null hypothesis, the two categorical variables are dependent
    # which means, the PerspectiveAPI's labels is credible for the toxicity evaluation for the HateXplain dataset

    return True


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

    # check overall scoring resutts by PerspectiveAPI OG and all 4 dialects
    print_results(
        og_scores, aave_scores, nigerianD_scores, indianD_scores, singlish_scores
    )

    # collect all error indexes by adding the offset
    to_drop2 = [i + 5000 for i in to_drop2]
    to_drop3 = [i + 10000 for i in to_drop3]
    to_drop4 = [i + 15000 for i in to_drop4]
    to_drop = sorted(to_drop1 + to_drop2 + to_drop3 + to_drop4, reverse=True)

    # perform statistical test to check the similarity between gold labels and PerspectiveAPI's labels
    check_perspective_credibility(hatexplain_df, og_scores, to_drop)
