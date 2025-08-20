import os
import urllib.request
from pathlib import Path
from typing import Literal

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


models = {
    "small": {
        "entity": "entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model",
        "coref": "coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model",
        "quote": "speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model",
    },
    "big": {
        "entity": "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model",
        "coref": "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model",
        "quote": "speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model",
    },
}

home = str(Path.home())
models_path = os.path.join(home, "booknlp_models")


def download_model(model_size: Literal["small", "big"]):
    if model_size not in models:
        raise ValueError(f"Model size {model_size} not recognized")

    def download(model_name: Literal["entity", "coref", "quote"]):
        modelPath = os.path.join(models_path, model_name)
        if not Path(modelPath).is_file():
            print(f"Downloading {model_name}")
            urllib.request.urlretrieve(
                f"http://people.ischool.berkeley.edu/~dbamman/booknlp_models/{model_name}", modelPath
            )
        elif Path(modelPath).is_file():
            print(f"Model {model_name} already downloaded")
        else:
            raise ValueError(f"Model {model_name} not found")

    for model_type, model_name in models[model_size].items():
        download(model_name)


if __name__ == "__main__":
    download_model("small")
