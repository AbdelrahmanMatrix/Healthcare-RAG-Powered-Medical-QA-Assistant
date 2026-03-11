import os
from datasets import load_dataset
import pandas as pd
from typing import Dict, Any

class MedicalDataLoader:
    def __init__(self):
        self.datasets_info = {
            "pubmedqa": "llamafactory/PubMedQA",
            "chatdoctor": "lavita/ChatDoctor-HealthCareMagic-100k",
            "medqa": "medalpaca/medical_meadow_medqa"
        }

    def load_pubmedqa(self) -> pd.DataFrame:
        print("Loading PubMedQA...")
        ds = load_dataset(self.datasets_info["pubmedqa"], split="train")
        return pd.DataFrame(ds)

    def load_chatdoctor(self) -> pd.DataFrame:
        print("Loading ChatDoctor...")
        ds = load_dataset(self.datasets_info["chatdoctor"], split="train")
        return pd.DataFrame(ds)

    def load_medqa(self) -> pd.DataFrame:
        print("Loading MedQA...")
        ds = load_dataset(self.datasets_info["medqa"], split="train")
        return pd.DataFrame(ds)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        return {
            "pubmedqa": self.load_pubmedqa(),
            "chatdoctor": self.load_chatdoctor(),
            "medqa": self.load_medqa()
        }

if __name__ == "__main__":
    loader = MedicalDataLoader()
    data = loader.load_all()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
