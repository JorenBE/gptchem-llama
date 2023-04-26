from typing import Optional
from .gptjclassifier import GPTJClassifier
from gptchem.extractor import ClassificationExtractor, RegressionExtractor
from gptchem.formatter import ClassificationFormatter, RegressionFormatter
from numpy.typing import ArrayLike
import numpy as np 
from gptjchem.gptj import create_dataloaders_from_frames, load_model, tokenizer, train
import gc
import torch 
from tqdm import tqdm
from more_itertools import chunked

class GPTRegressor(GPTJClassifier):
    """Approximate regression by predicting rounded floats"""
    
    def __init__(self, 
        property_name: str,
        extractor: RegressionExtractor = RegressionExtractor(),
        batch_size: int = 4,
        tune_settings: Optional[dict] = None,
        inference_batch_size: int = 4,
          inference_max_new_tokens: int = 200,
        num_digits: int = 2):
        """Initialize a GPT-J based regressor

        Arg: 
            property_name (str): Name of the property to be predicted
            extractor (RegressionExtractor): Extractor for the property
            batch_size (int, optional): Batch size for fine-tuning. Defaults to 4.
            tune_settings (Optional[dict], optional): Settings for fine-tuning. Defaults to None.
            inference_batch_size (int, optional): Batch size for inference. Defaults to 4.
            inference_max_new_tokens (int, optional): Maximum number of tokens to generate during inference. Defaults to 200.
            num_digits (int): Number of digits the completions will be rounded to.
        """
        self.property_name = property_name
        self.extractor = extractor
        self.batch_size = batch_size
        self.tune_settings = tune_settings or {}
        self.inference_batch_size = inference_batch_size
        self.inference_max_new_tokens = inference_max_new_tokens

        self.formatter = RegressionFormatter(
            representation_column="repr",
            label_column="prop",
            property_name=property_name,
            num_digits=num_digits,
        )
        self.model = load_model()

class BinnedGPTJRegressor(GPTJClassifier):
    """Wrapper around GPT-3 for "regression"
    by binning the property values in sufficiently many bins.

    The predicted property values are the bin centers.
    """
    def __init__(
        self,
        property_name: str,
        querier_settings: Optional[dict] = None,
        extractor: ClassificationExtractor = ClassificationExtractor(),
        desired_accuracy: float = 1.0,
        batch_size: int = 4,
        tune_settings: Optional[dict] = None,
        inference_batch_size: int = 4,
        inference_max_new_tokens: int = 200,
        equal_bin_sizes: bool = True,
        device=None
    ):
        self.property_name = property_name
        self.querier_settings = querier_settings
        self.extractor = extractor
        self.desired_accuracy = desired_accuracy
        self.batch_size = batch_size
        self.tune_settings = tune_settings or {}
        self.inference_batch_size = inference_batch_size
        self.inference_max_new_tokens = inference_max_new_tokens
        self.model = load_model(device=device)
        self.equal_bin_sizes = equal_bin_sizes
    
    def _fit(self, formatted):
        dl = create_dataloaders_from_frames(formatted, None, batch_size=self.batch_size)
        train(self.model, dl["train"], **self.tune_settings)
        dl = None
        formatted = None
        gc.collect()

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Array of molecular representations.
            y (ArrayLike): Array of property values.
        """
        # set num bins
        num_bins = int(np.ceil((max(y) - min(y)) / self.desired_accuracy))

        # set formatter
        self.formatter = ClassificationFormatter(
            representation_column="repr",
            label_column="prop",
            property_name=self.property_name,
            num_classes=num_bins,
            qcut=not self.equal_bin_sizes,
        )

        df = self._prepare_df(X, y)
        formatted = self.formatter(df)
        self._fit(formatted)

    def bin_indices_to_ranges(self, predicted_bin_indices: ArrayLike):
        """Convert a list of predicted bin indices to a list of bin ranges

        Use the bin edges from self.formatter.bins

        Args:
            predicted_bin_indices (ArrayLike): List of predicted bin indices

        Returns:
            ArrayLike: List of bin range tuples
        """
        bin_ranges = []
        for bin_index in predicted_bin_indices:
            bin_ranges.append((self.formatter.bins[bin_index], self.formatter.bins[bin_index + 1]))
        return bin_ranges

    def predict(self, X: ArrayLike, remap: bool = True, temperature: float=0, do_sample:bool=False) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Array of molecular representations.
            remap (bool, optional): Whether to remap the predicted bin indices to the
            temperature (float, optional): Temperature for sampling. Defaults to 0.

        Returns:
            ArrayLike: Predicted property values
        """
        df = self._prepare_df(X, [0] * len(X))
        formatted = self.formatter(df)
        completions = []

        self.model.eval()
        device = self.model.device
        with torch.no_grad():
            for chunk in tqdm(
                chunked(range(len(formatted)), self.inference_batch_size),
                total=len(formatted) // self.inference_batch_size,
            ):
                rows = formatted.iloc[chunk]
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
            self.extractor.extract(completions[i].split("###")[1]) for i in range(len(completions))
        ]
        extracted = np.array(extracted, dtype=int)
        if remap:
            # self.formatter.bins is the list of bin edges
            # we want to remap the bin indices to the bin centers
            # so we take the average of each bin edge with the next one
            # for the first and last bin we just take the right or left edge, respectively
            centers = [self.formatter.bins[1]]
            for i in range(1, len(self.formatter.bins) - 2):
                centers.append((self.formatter.bins[i] + self.formatter.bins[i + 1]) / 2)
            centers.append(self.formatter.bins[-2])

            extracted = [centers[i] for i in extracted]

        return extracted

