import os

import pandas as pd
from fastcore.xtras import save_pickle
from gptchem.data import get_esol_data, get_solubility_test_data
from gptchem.evaluator import evaluate_classification
from gptchem.utils import make_outdir
from sklearn.model_selection import train_test_split
import numpy as np
from gptjchem.gptjclassifier import GPTJClassifier

REPRESENTATIONS = [
    'SMILES',
    'name', 
    'selfies',
][::-1]

MAX_NUM_TEST_POINTS = 250

def train_test(size, representation, seed, num_epochs):
    data = get_esol_data()
    data = data.dropna(subset=["measured log(solubility:mol/L)"])
    train_subset = data.sample(n=size, random_state=seed)
    train_subset = train_subset.reset_index(drop=True)
    train_subset["binned"] = (
        train_subset["measured log(solubility:mol/L)"]
        > np.median(data["measured log(solubility:mol/L)"])
    ).astype(int)
    test = get_solubility_test_data()
    test["binned"] = (
        test["measured log(solubility:mol/L)"] > np.median(data["measured log(solubility:mol/L)"])
    ).astype(int)

    classifier = GPTJClassifier("solubility", tune_settings={"num_epochs": num_epochs, "lr": 1e-4}, inference_batch_size=2, inference_max_new_tokens=100)
    
    classifier.fit(train_subset[representation].to_list(), train_subset["binned"].to_list())

    y_pred = classifier.predict(test[representation].to_list())

    results = evaluate_classification(test["binned"].to_list(), y_pred)

    dirname = make_outdir('')

    res  = {
        **results,
        "size": size,
        "representation": representation,
        "num_epochs": num_epochs,
        "seed": seed,
    }
    save_pickle(os.path.join(dirname, f"results_{size}_{seed}_{representation}.pkl"), res)


if __name__ == "__main__":
    for seed in range(5):
        for representation in REPRESENTATIONS:
            for size in [10, 20, 50, 100, 200][::-1]:
                for num_epochs in [4, 8, 14, 20]:
                    try:
                        train_test(size, representation, seed + 14556, num_epochs)
                    except Exception as e:
                        print(e)
                        pass
