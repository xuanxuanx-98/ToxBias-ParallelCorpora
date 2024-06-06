"""Run Perspective API on all text instances
Save the scores to csv and error indices of each batch as json (simple list)

Prequisites:
    - a Google Perspective API key
    - the original HateXplain dataset in json format
    - converted dialect datasets in jsonl format

Usage:
    $ python retrievePerspectiveScores.py
"""

import pandas as pd
import json
import time
from tqdm import tqdm

from googleapiclient import discovery


def get_persp_prediction(text):
    """Load API key and get the prediction for a single text instance"""
    API_KEY = None  # this is a placeholder, replace with your own API key

    client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
    )

    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()

    return response


def run_batch_on_og(df_batch, n_batch):
    """Run Perspective API on the original HateXplain dataset in one batch"""
    error_instances = []
    scores = []

    # get scores for one text instance at a time
    for i in tqdm(range(len(df_batch))):
        text = " ".join(list(df_batch["post_tokens"].iloc[i]))

        try:
            res = get_persp_prediction(text)
        except Exception as e:
            error_instances.append(i)
            print(f"Error at index {i}: {e}")
            continue

        score = res["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

        scores.append(score)
        # print(scores)
        time.sleep(0.8)  # pause for 0.8 second to avoid rate limit

    scores_df = pd.DataFrame(scores, columns=['score'])
    scores_df.to_csv(f"../scores/persp_score_original_batch{n_batch}.csv", sep=",", index=False)

    with open(f"../scores/errors_original_batch{n_batch}.json", 'w') as f:
        json.dump(error_instances, f)

    return True


def run_batch_on_dialect(df_batch, dialect, n_batch):
    """Run Perspective API on any converted dialect dataset in one batch"""
    error_instances = []
    scores = []

    # get scores for one text instance at a time
    for i in tqdm(range(len(df_batch))):
        text = df_batch["text"][i]

        try:
            res = get_persp_prediction(text)
        except Exception as e:
            error_instances.append(i)
            print(f"Error at index {i}: {e}")
            continue

        score = res["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

        scores.append(score)
        # print(scores)
        time.sleep(0.8)  # pause for 0.8 second to avoid rate limit

    scores_df = pd.DataFrame(scores, columns=['score'])
    scores_df.to_csv(f"../scores/persp_score_{dialect}_batch{n_batch}.csv", sep=",", index=False)

    with open(f"../scores/errors_{dialect}_batch{n_batch}.json", 'w') as f:
        json.dump(error_instances, f)

    return True


if __name__ == "__main__":
    # read in original data
    hatexplain_df = pd.read_json(f"../data/hatexplain_original.json").transpose()
    # run Perspective API on the original data in 4 batches
    hatexplain_df_batch1 = hatexplain_df[:5000]
    run_batch_on_og(hatexplain_df_batch1, "1")
    hatexplain_df_batch2 = hatexplain_df[5000:10000]
    run_batch_on_og(hatexplain_df_batch2, "2")
    hatexplain_df_batch3 = hatexplain_df[10000:15000]
    run_batch_on_og(hatexplain_df_batch3, "3")
    hatexplain_df_batch4 = hatexplain_df[15000:]
    run_batch_on_og(hatexplain_df_batch4, "4")

    # do the same for all 4 dialects data, but with different function
    aave_full = pd.read_json("../data/aave_full.jsonl", lines=True)
    aave_batch1 = aave_full[:5000].reset_index(drop=True)
    run_batch_on_dialect(aave_batch1, "aave", "1")
    aave_batch2 = aave_full[5000:10000].reset_index(drop=True)
    run_batch_on_dialect(aave_batch2, "aave", "2")
    aave_batch3 = aave_full[10000:15000].reset_index(drop=True)
    run_batch_on_dialect(aave_batch3, "aave", "3")
    aave_batch4 = aave_full[15000:].reset_index(drop=True)
    run_batch_on_dialect(aave_batch4, "aave", "4")

    nigerianD_full = pd.read_json("../data/nigerianD_full.jsonl", lines=True)
    nigerianD_batch1 = nigerianD_full[:5000].reset_index(drop=True)
    run_batch_on_dialect(nigerianD_batch1, "nigerianD", "1")
    nigerianD_batch2 = nigerianD_full[5000:10000].reset_index(drop=True)
    run_batch_on_dialect(nigerianD_batch2, "nigerianD", "2")
    nigerianD_batch3 = nigerianD_full[10000:15000].reset_index(drop=True)
    run_batch_on_dialect(nigerianD_batch3, "nigerianD", "3")
    nigerianD_batch4 = nigerianD_full[15000:].reset_index(drop=True)
    run_batch_on_dialect(nigerianD_batch4, "nigerianD", "4")

    indianD_full = pd.read_json("../data/indianD_full.jsonl", lines=True)
    indianD_batch1 = indianD_full[:5000].reset_index(drop=True)
    run_batch_on_dialect(indianD_batch1, "indianD", "1")
    indianD_batch2 = indianD_full[5000:10000].reset_index(drop=True)
    run_batch_on_dialect(indianD_batch2, "indianD", "2")
    indianD_batch3 = indianD_full[10000:15000].reset_index(drop=True)
    run_batch_on_dialect(indianD_batch3, "indianD", "3")
    indianD_batch4 = indianD_full[15000:].reset_index(drop=True)
    run_batch_on_dialect(indianD_batch4, "indianD", "4")

    singlish_full = pd.read_json("../data/singlish_full.jsonl", lines=True)
    singlish_batch1 = singlish_full[:5000].reset_index(drop=True)
    run_batch_on_dialect(singlish_batch1, "singlish", "1")
    singlish_batch2 = singlish_full[5000:10000].reset_index(drop=True)
    run_batch_on_dialect(singlish_batch2, "singlish", "2")
    singlish_batch3 = singlish_full[10000:15000].reset_index(drop=True)
    run_batch_on_dialect(singlish_batch3, "singlish", "3")
    singlish_batch4 = singlish_full[15000:].reset_index(drop=True)
    run_batch_on_dialect(singlish_batch4, "singlish", "4")

