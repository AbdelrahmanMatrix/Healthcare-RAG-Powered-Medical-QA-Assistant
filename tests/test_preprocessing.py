import unittest
import pandas as pd
from src.data.preprocessor import MedicalPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = MedicalPreprocessor()

    def test_clean_text(self):
        text = "Hello  \n world "
        self.assertEqual(self.preprocessor.clean_text(text), "Hello world")

    def test_preprocess_pubmedqa(self):
        df = pd.DataFrame({
            'instruction': ['Answer the question based on the following context: Context'],
            'input': ['Question: My Question'],
            'output': ['Answer']
        })
        processed = self.preprocessor.preprocess_pubmedqa(df)
        self.assertEqual(processed.iloc[0]['question'], 'My Question')
        self.assertEqual(processed.iloc[0]['context'], 'Context')

if __name__ == '__main__':
    unittest.main()
