from typing import Optional
from .gptjclassifier import GPTJClassifier
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter

class BinnedGPTJRegressor(GPTJClassifier):
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
    ):
        ...