import os

import pandas as pd
from fastcore.xtras import save_pickle
from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classification
from gptchem.utils import make_outdir
from sklearn.model_selection import train_test_split

from gptjchem.gptjclassifier import GPTJClassifier

REPRESENTATIONS = [
    'SMILES',
    'name', 
    'selfies',
][::-1]

MAX_NUM_TEST_POINTS = 100

def train_test(size, representation, seed, num_epochs):
    df = get_photoswitch_data()
    df = df.dropna(subset=[representation, 'E isomer pi-pi* wavelength in nm'])
    df['y'] = pd.qcut(df['E isomer pi-pi* wavelength in nm'], 2, labels=[0,1])
    train_df, test_df = train_test_split(df, train_size=size, test_size=min([len(df)-size, MAX_NUM_TEST_POINTS]), random_state=seed)

    classifier = GPTJClassifier("transition wavelength", tune_settings={"num_epochs": num_epochs, "lr": 1e-4}, inference_batch_size=2, inference_max_new_tokens=100)
    
    classifier.fit(train_df["SMILES"].to_list(), train_df["y"].to_list())

    y_pred = classifier.predict(test_df["SMILES"].to_list())

    results = evaluate_classification(test_df["y"].to_list(), y_pred)

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
        for num_epochs in [4, 8, 14, 20]:
            for representation in REPRESENTATIONS:
                for size in [10, 20, 50, 100, 200][::-1]:
                    try:
                        train_test(size, representation, seed + 14556, num_epochs)
                    except Exception as e:
                        print(e)
                        pass
