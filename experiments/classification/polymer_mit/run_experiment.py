import os

import pandas as pd
from fastcore.xtras import save_pickle
from gptchem.evaluator import evaluate_classification
from gptchem.utils import make_outdir
from sklearn.model_selection import train_test_split
import numpy as np
from gptjchem.gptjclassifier import GPTJClassifier


MAX_NUM_TEST_POINTS = 500
num_classes = [2, 5, 10]

def train_test(size, num_classes: int, seed, num_epochs):
    data = pd.read_csv('data.csv')

    data['binned'] = pd.qcut(data['y'], num_classes, labels=False)

    train, test = train_test_split(data, train_size=size, test_size=min(len(data)-size, MAX_NUM_TEST_POINTS), random_state=seed, stratify=data['binned'])
    classifier = GPTJClassifier("adhesive free energy", tune_settings={"num_epochs": num_epochs, "lr": 1e-4}, inference_batch_size=2, inference_max_new_tokens=100)
    
    classifier.fit(train["SMILES"].to_list(), train["binned"].to_list())

    y_pred = classifier.predict(test["SMILES"].to_list())

    results = evaluate_classification(test["binned"].to_list(), y_pred)

    dirname = make_outdir('')

    res  = {
        **results,
        "size": size,
        "num_epochs": num_epochs,
        "seed": seed,
    }
    save_pickle(os.path.join(dirname, f"results_{size}_{seed}_{num_classes}.pkl"), res)


if __name__ == "__main__":
    for num_epochs in [4, 8, 14, 20]:
        for seed in range(5):
            for size in [10,  50, 100, 500, 700, 5000, 10000][::-1]:
                for num_class in num_classes:
                    try:
                        train_test(size, num_class, seed + 14556, num_epochs)
                    except Exception as e:
                        print(e)
                        pass
