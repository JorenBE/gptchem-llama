import os

import pandas as pd
from fastcore.xtras import save_pickle
from gptchem.evaluator import get_regression_metrics
from gptchem.utils import make_outdir
from sklearn.model_selection import train_test_split
import numpy as np
from gptjchem.gptjregressor import BinnedGPTJRegressor


MAX_NUM_TEST_POINTS = 250
accuracies = [0.1, 0.5, 1.0, 2.0, 5.0]

def train_test(size, accuracy: float, seed, num_epochs):
    data = pd.read_csv('train - Dennis Svatunek.csv')

    data['binned'] = pd.qcut(data['Free energy of activation (kcal/mol)'], 5, labels=False)

    train, test = train_test_split(data, train_size=size, test_size=min(len(data)-size, MAX_NUM_TEST_POINTS), random_state=seed, stratify=data['binned'])
    model = BinnedGPTJRegressor("free energy of activation", tune_settings={"num_epochs": num_epochs, "lr": 1e-4}, inference_batch_size=2, inference_max_new_tokens=100, desired_accuracy=accuracy, equal_bin_sizes=False)
    
    model.fit(train["SMILES"].to_list(), train["y"].to_list())

    y_pred = model.predict(test["SMILES"].to_list())

    results = get_regression_metrics(test["y"].to_list(), y_pred)

    dirname = make_outdir('')

    res  = {
        **results,
        "size": size,
        "num_epochs": num_epochs,
        "seed": seed,
    }
    save_pickle(os.path.join(dirname, f"results_regression_{size}_{seed}_{accuracy}.pkl"), res)


if __name__ == "__main__":
    for seed in range(5):
        for num_epochs in [20]:
            for size in [10,  50, 100, 500, 700]:
                for acc in accuracies:
                    #try:
                    train_test(size, acc, seed + 14556, num_epochs)
                    # except Exception as e:
                    #     print(e)
                    #     pass
