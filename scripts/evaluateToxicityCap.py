"""Check the toxicity scores by Perspective API for the original and 4 dialect text data

Question:
    - How does the toxicity score change when the original HateXplain dataset is converted to 4 different dialects?
    - Split the gold toxic and non-toxic: is there a cap on gold toxic instances? I.e.:
        - if the original text is already toxic, the dialect text could hardly be more toxic
        - if the original text is non-toxic, the dialect text could get more toxic more easily due to various dialect specific features

Usage:
    $ python evaluateToxicityCap.py
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


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
    Return: tuple of two lists (scores of gold toxic instances, scores of gold non-toxic instances)"""
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


def print_tox_increase_count(og_splits, dialect_splits, dialect_name):
    """Print the count of instances where the dialect scores are higher than the original scores
    for both gold toxic and non-toxic sub-sets separately"""
    print(f"Dialect in test: {dialect_name}")

    # condition 1: instances that are gold toxic
    gtox_og_scores = og_splits[0]
    gtox_dialect_scores = dialect_splits[0]
    gtox_inc = [(a, b) for a, b in zip(gtox_og_scores, gtox_dialect_scores) if a < b]
    print("gold toxic total:    ", len(gtox_og_scores), "dialect>og:", len(gtox_inc), "--->", round(len(gtox_inc)/len(gtox_og_scores), 4))

    # condition 2: instances that are gold non-toxic
    gntox_og_scores = og_splits[1]
    gntox_dialect_scores = dialect_splits[1]
    gntox_inc = [(a, b) for a, b in zip(gntox_og_scores, gntox_dialect_scores) if a < b]
    print("gold non-toxic total:", len(gntox_og_scores), " dialect>og:", len(gntox_inc), "--->", round(len(gntox_inc)/len(gntox_og_scores), 4))

    return True


def save_all_score_plots(og_splits, aave_splits, nigerianD_splits, indianD_splits, singlish_splits):
    """Save boxplots of all scores across original and dialects
    The scores are split into two subplots based on gold labels"""
    gtox_og_scores = og_splits[0]
    gtox_aave_scores = aave_splits[0]
    gtox_nigeriand_scores = nigerianD_splits[0]
    gtox_indiand_scores = indianD_splits[0]
    gtox_singlish_scores = singlish_splits[0]

    gntox_og_scores = og_splits[1]
    gntox_aave_scores = aave_splits[1]
    gntox_nigeriand_scores = nigerianD_splits[1]
    gntox_indiand_scores = indianD_splits[1]
    gntox_singlish_scores = singlish_splits[1]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.boxplot([gntox_og_scores, gntox_aave_scores, gntox_nigeriand_scores, gntox_indiand_scores, gntox_singlish_scores], labels=["Original", "AAVE", "NigerianD", "IndianD", "Singlish"]);
    plt.title("Perspective scores of gold non-toxic texts")

    plt.subplot(1, 2, 2)
    plt.boxplot([gtox_og_scores, gtox_aave_scores, gtox_nigeriand_scores, gtox_indiand_scores, gtox_singlish_scores], labels=["Original", "AAVE", "NigerianD", "IndianD", "Singlish"]);
    plt.title("Perspective scores of gold toxic texts")

    # save plot for overviewing all scores across original and dialects 
    plt.savefig("../figures/all-scores.png")

    return True


def save_score_change_plots(og_splits, dialect_splits, dialect_name):
    """Calculate quartiles for each set of scores and create boxplots
    Colored Lines indicate score changes for each instance
    The scores are split into two subplots based on gold labels"""
    gtox_og_scores = og_splits[0]
    gtox_dialect_scores = dialect_splits[0]
    gntox_og_scores = og_splits[1]
    gntox_dialect_scores = dialect_splits[1]

    # create a figure and two subplots with a larger height
    fig, ax = plt.subplots(ncols=2, figsize=(15, 10))  # adjust the first value to increase the width
    # calculate quartiles for each set of scores
    q1_gntox_og, q3_gntox_og = np.percentile(gntox_og_scores, [25, 75])
    q1_gntox_dialect, q3_gntox_dialect = np.percentile(gntox_dialect_scores, [25, 75])
    q1_gtox_og, q3_gtox_og = np.percentile(gtox_og_scores, [25, 75])
    q1_gtox_dialect, q3_gtox_dialect = np.percentile(gtox_dialect_scores, [25, 75])

    # create boxplots and lines for original scores and dialect scores
    ax[0].boxplot([gntox_og_scores, gntox_dialect_scores], positions=[1, 2], widths=0.6, medianprops={"color":"slategrey"})
    for i in range(len(gntox_og_scores)):
        if q1_gntox_og <= gntox_og_scores[i] <= q3_gntox_og and q1_gntox_dialect <= gntox_dialect_scores[i] <= q3_gntox_dialect:
            if gntox_og_scores[i] > gntox_dialect_scores[i]:
                color = "lightblue"
            else:
                color = "darkorange"
            ax[0].plot([1, 2], [gntox_og_scores[i], gntox_dialect_scores[i]], color=color, linestyle="-", linewidth=1)
    ax[0].set_xticks([1, 2])
    ax[0].set_xticklabels(["Original", f"{dialect_name} (converted)"])
    ax[0].set_title("Perspective scores of gold non-toxic texts")

    # create boxplots and lines for original scores and dialect scores
    ax[1].boxplot([gtox_og_scores, gtox_dialect_scores], positions=[1, 2], widths=0.6, medianprops={"color":"slategrey"})
    for i in range(len(gtox_og_scores)):
        if q1_gtox_og <= gtox_og_scores[i] <= q3_gtox_og and q1_gtox_dialect <= gtox_dialect_scores[i] <= q3_gtox_dialect:
            if gtox_og_scores[i] > gtox_dialect_scores[i]:
                color = "lightblue"
            else:
                color = "darkorange"
            ax[1].plot([1, 2], [gtox_og_scores[i], gtox_dialect_scores[i]], color=color, linestyle="-", linewidth=1)
    ax[1].set_xticks([1, 2])
    ax[1].set_xticklabels(["Original", f"{dialect_name} (converted)"])
    ax[1].set_title("Perspective scores of gold toxic texts")

    # save the plot to the figures folder
    plt.savefig(f"../figures/{dialect_name}-changes.png")
    print(f"|-- {dialect_name} done!")

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

    # print out the count of instances where the dialect scores are higher than the original scores
    print_tox_increase_count(og_splits, aave_splits, "AAVE")
    print_tox_increase_count(og_splits, nigerianD_splits, "NigerianD")
    print_tox_increase_count(og_splits, indianD_splits, "IndianD")
    print_tox_increase_count(og_splits, singlish_splits, "Singlish")

    # save boxplots of all scores across original and 4 dialects
    print("\nSaving all scores plot (split by gold toxic/non-toxc) ...", end="", flush=True)
    save_all_score_plots(og_splits, aave_splits, nigerianD_splits, indianD_splits, singlish_splits)
    print(" done!")

    # save boxplots of score changes of each instance for each dialect
    print("Saving OG scores vs. dialect scores comparison plots ...")
    save_score_change_plots(og_splits, aave_splits, "AAVE")
    save_score_change_plots(og_splits, nigerianD_splits, "NigerianD")
    save_score_change_plots(og_splits, indianD_splits, "IndianD")
    save_score_change_plots(og_splits, singlish_splits, "Singlish")
    print("All plots saved successfully!")






