import gc
from typing import List, Optional
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.gpt_classifier import GPTClassifier
from more_itertools import chunked
from numpy.typing import ArrayLike
from tqdm import tqdm

from gptjchem.gptj import create_dataloaders_from_frames, load_model, tokenizer, train


class GPTJClassifier(GPTClassifier):
    def __init__(
        self,
        property_name: str,
        querier_settings: Optional[dict] = None,
        extractor: ClassificationExtractor = ClassificationExtractor(),
        batch_size: int = 4,
        tune_settings: Optional[dict] = None,
        inference_batch_size: int = 4,
        inference_max_new_tokens: int = 200,
        device: str = "cuda",
        formatter: Optional[ClassificationFormatter] = None,
        representation_names: Optional[List[str]] = None,
    ):
        self.property_name = property_name
        self.querier_settings = querier_settings
        self.extractor = extractor
        self.batch_size = batch_size
        self.tune_settings = tune_settings or {}
        self.inference_batch_size = inference_batch_size
        self.inference_max_new_tokens = inference_max_new_tokens
        self.device = device

        self.formatter = (
            ClassificationFormatter(
                representation_column="repr",
                label_column="prop",
                property_name=property_name,
                num_classes=None,
            )
            if formatter is None
            else formatter
        )
        self.model = load_model(device=device)
        self.representation_names = representation_names if representation_names else []

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def fit(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        formatted: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            y (ArrayLike): Target data (typically array of property values)
            formatted (pd.DataFrame): Formatted data (typically output of `formatter`)
        """
        if formatted is None:
            if X is None or y is None:
                raise ValueError("Either formatted data or X and y must be provided.")

        X = np.array(X)
        y = np.array(y)
        if formatted is None:
            if X.ndim == 1 or (X.ndim == 2 and X.size == len(X)):
                df = self._prepare_df(X, y)
                formatted = self.formatter(df)
            elif X.ndim == 2 and X.size > len(X):
                if not len(self.representation_names) == X.shape[1]:
                    raise ValueError(
                        "Number of representation names must match number of dimensions"
                    )

                dfs = []
                for i in range(X.ndim):
                    formatter = deepcopy(self.formatter)
                    formatter.representation_name = self.representation_names[i]
                    df = self._prepare_df(X[:, i], y)
                    formatted = formatter(df)
                    dfs.append(formatted)

                formatted = pd.concat(dfs)

        dl = create_dataloaders_from_frames(formatted, None, batch_size=self.batch_size)
        train(self.model, dl["train"], **self.tune_settings)
        dl = None
        gc.collect()

    def _predict(
        self,
        X: Optional[ArrayLike] = None,
        temperature=0.7,
        do_sample=False,
        formatted: Optional[pd.DataFrame] = None,
    ) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            temperature (float, optional): Temperature for sampling. Defaults to 0.7.
            do_sample (bool, optional): Whether to sample or not. Defaults to False.
            formatted (pd.DataFrame, optional): Formatted data (typically output of `formatter`).
                Defaults to None. If None, X must be provided.

        Returns:
            ArrayLike: Predicted property values
        """

        if formatted is None:
            if X is None:
                raise ValueError("Either formatted data or X must be provided.")

        if formatted is None:
            if X.ndim == 1 or (X.ndim == 2 and X.size == len(X)):
                df = self._prepare_df(X, [0] * len(X))
                formatted = self.formatter(df)
                dfs = [formatted]
            elif X.ndim == 2 and X.size > len(X):
                if not len(self.representation_names) == X.shape[1]:
                    raise ValueError(
                        "Number of representation names must match number of dimensions"
                    )

                dfs = []
                for i in range(X.shape[1]):
                    formatter = deepcopy(self.formatter)
                    formatter.representation_name = self.representation_names[i]
                    df = self._prepare_df(X[:, i], [0] * len(X))
                    formatted = formatter(df)
                    dfs.append(formatted)

        else:
            dfs = [formatted]

        predictions = []
        for df in dfs:
            predictions.append(self._query(df, temperature=temperature, do_sample=do_sample))

        return predictions

    def predict(
        self,
        X: Optional[ArrayLike] = None,
        temperature=0.7,
        do_sample=False,
        formatted: Optional[pd.DataFrame] = None,
        return_std: bool = True,
    ):
        predictions = self._predict(
            X=X, temperature=temperature, do_sample=do_sample, formatted=formatted
        )

        predictions = np.array(predictions).T

        predictions_mode = np.array([np.argmax(np.bincount(pred)) for pred in predictions.astype(int)])
        
        if return_std:
            predictions_std = np.array([np.std(pred) for pred in predictions.astype(int)])
            return predictions_mode, predictions_std
        return predictions_mode


    def _query(self, formatted_df, temperature, do_sample):
        completions = []
        self.model.eval()
        device = self.model.device
        with torch.no_grad():
            for chunk in tqdm(
                chunked(range(len(formatted_df)), self.inference_batch_size),
                total=len(formatted_df) // self.inference_batch_size,
            ):
                rows = formatted_df.iloc[chunk]
                prompt = tokenizer(
                    rows["prompt"].to_list(),
                    truncation=False,
                    padding=True,
                    max_length=self.inference_max_new_tokens,
                    return_tensors="pt",
                )
                prompt = {key: value.to(device) for key, value in prompt.items()}
                out = self.model.generate(
                    **prompt,
                    temperature=temperature,
                    max_new_tokens=self.inference_max_new_tokens,
                    do_sample=do_sample,
                )
                completions.extend([tokenizer.decode(out[i]) for i in range(len(out))])

        extracted = [
            self.extractor.extract(completions[i].split("###")[1])
            for i in range(
                len(completions)
            )  # ToDo: Make it possible to use other splitters than ###
        ]

        filtered = [v if v is not None else np.nan for v in extracted]

        return extracted
