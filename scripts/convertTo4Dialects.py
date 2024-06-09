"""Convert the HateXplain dataset to 4 different dialects: AAVE, Nigerian, Indian, and Singlish

Prequisites:
- clone multi-value repo and switch to version bf1aea58303ea70d8d380294f97886d821a940a2
    - git clone git@github.com:SALT-NLP/multi-value.git
    - git checkout bf1aea58303ea70d8d380294f97886d821a940a2 
- install requirements from REQUIREMENTS_MultiV.txt
- move this script and the HateXplain dataset (../data/hatexplain_original.json) to the multi-value root directory

- Usage:
    from the multi-value root directory run:
    $ python convertTo4Dialects.py
"""

import pandas as pd
import json
from tqdm import tqdm

from src.Dialects import AfricanAmericanVernacular
from src.Dialects import NigerianDialect
from src.Dialects import ColloquialSingaporeDialect
from src.Dialects import IndianDialect


def transform_to_dialect(dialect, df, dialect_name):
    """Take one dialect transform module and apply it to the HateXplain dataset
    Save the results in a jsonl file"""
    sents = []  # {text: ..., rules: [...]}

    for i in tqdm(range(len(df)), desc="Processing"):
        sent = " ".join(df["post_tokens"][0])  # load original sentece

        sent_dict = {}
        sent_dict["text"] = dialect.convert_sae_to_dialect(sent)
        sent_dict["rules"] = list(
            set([i["type"] for i in dialect.executed_rules.values()])
        )

        sents.append(sent_dict)

    with open(f"{dialect_name}.jsonl", "w") as outfile:
        for entry in sents:
            json.dump(entry, outfile)
            outfile.write("\n")

    return True


if __name__ == "__main__":
    # read in the original HateXplain dataset
    df = pd.read_json(f"./hatexplain_original.json").transpose()

    # load and run AAVE transform module, save results
    aave = AfricanAmericanVernacular()
    transform_to_dialect(dialect=aave, df=df, dialect_name="aave")
    # load and run Nigerian dialect transform module, save results
    ngd = NigerianDialect()
    transform_to_dialect(dialect=ngd, df=df, dialect_name="nigerianD")
    # load and run HongKong dialect transform module, save results
    indd = IndianDialect()
    transform_to_dialect(dialect=indd, df=df, dialect_name="indianD")
    # load and run Singlish dialect transform module, save results
    csgd = ColloquialSingaporeDialect()
    transform_to_dialect(dialect=csgd, df=df, dialect_name="singlish")
