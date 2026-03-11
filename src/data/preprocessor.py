import pandas as pd
import re
from typing import List

class MedicalPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Remove multiple newlines and spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def preprocess_pubmedqa(self, df: pd.DataFrame) -> pd.DataFrame:
        # PubMedQA has 'instruction' (context), 'input' (question), 'output' (answer)
        processed_df = pd.DataFrame()
        processed_df['question'] = df['input'].str.replace('Question: ', '', regex=False)
        processed_df['context'] = df['instruction'].str.replace('Answer the question based on the following context: ', '', regex=False)
        processed_df['answer'] = df['output']
        processed_df['source'] = 'pubmedqa'
        return processed_df

    def preprocess_chatdoctor(self, df: pd.DataFrame) -> pd.DataFrame:
        # ChatDoctor has 'instruction', 'input', 'output'
        processed_df = pd.DataFrame()
        processed_df['question'] = df['input']
        processed_df['context'] = ""
        processed_df['answer'] = df['output']
        processed_df['source'] = 'chatdoctor'
        return processed_df

    def preprocess_medqa(self, df: pd.DataFrame) -> pd.DataFrame:
        # MedQA has 'input', 'instruction', 'output'
        processed_df = pd.DataFrame()
        processed_df['question'] = df['input']
        processed_df['context'] = ""
        processed_df['answer'] = df['output']
        processed_df['source'] = 'medqa'
        return processed_df

    def unify_datasets(self, datasets: dict) -> pd.DataFrame:
        pubmed = self.preprocess_pubmedqa(datasets['pubmedqa'])
        chatdoctor = self.preprocess_chatdoctor(datasets['chatdoctor'])
        medqa = self.preprocess_medqa(datasets['medqa'])

        combined_df = pd.concat([pubmed, chatdoctor, medqa], ignore_index=True)

        # Clean all text columns
        for col in ['question', 'context', 'answer']:
            combined_df[col] = combined_df[col].apply(self.clean_text)

        # Drop duplicates
        combined_df = combined_df.drop_duplicates(subset=['question', 'answer'])

        return combined_df

if __name__ == "__main__":
    # Test with dummy data
    preprocessor = MedicalPreprocessor()
    dummy_pubmed = pd.DataFrame({
        'instruction': ['Answer the question based on the following context: Context here'],
        'input': ['Question: Question here?'],
        'output': ['Answer here']
    })
    processed = preprocessor.preprocess_pubmedqa(dummy_pubmed)
    print(processed)
