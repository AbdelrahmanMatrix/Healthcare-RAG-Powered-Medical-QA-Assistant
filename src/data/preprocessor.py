from pathlib import Path
import re

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect


# Keep language detection deterministic across runs.
DetectorFactory.seed = 0


RAW_COLUMNS = ["instruction", "input", "output"]
TARGET_COLUMNS = ["question", "context", "answer"]


def get_context(text: str) -> str:
	match = re.search(r"context:\s*(.*)", str(text), re.IGNORECASE | re.DOTALL)
	if match:
		return match.group(1).strip()
	return str(text).strip()


def get_question(text: str) -> str:
	match = re.search(r"question:\s*(.*)", str(text), re.IGNORECASE | re.DOTALL)
	if match:
		return match.group(1).strip()
	return str(text).strip()


def normalize_text(text: str) -> str:
	text = str(text)
	text = text.lower()
	text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
	text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove special characters
	text = re.sub(r"\s+", " ", text).strip()  # collapse and trim whitespace
	return text


def is_english_row(row: pd.Series) -> bool:
	combined = " ".join(str(row[col]) for col in TARGET_COLUMNS if col in row).strip()
	if not combined:
		return False

	try:
		return detect(combined) == "en"
	except LangDetectException:
		return False


def main() -> None:
	repo_root = Path(__file__).resolve().parents[2]
	raw_path = repo_root / "data" / "raw" / "pubmedqa_raw.csv"
	out_dir = repo_root / "data" / "processed"
	out_path = out_dir / "pubmedqa_cleaned.csv"

	df = pd.read_csv(raw_path)

	missing = [col for col in RAW_COLUMNS if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	rows_initial = len(df)
	print(f"Rows before preprocessing: {rows_initial}")

	# Extract clean semantic fields from raw columns.
	df["question"] = df["input"].fillna("").astype(str).apply(get_question)
	df["context"] = df["instruction"].fillna("").astype(str).apply(get_context)
	df["answer"] = df["output"].fillna("").astype(str)
	df = df[TARGET_COLUMNS].copy()

	# Apply text cleaning to all required columns.
	for col in TARGET_COLUMNS:
		df[col] = df[col].fillna("").astype(str).apply(normalize_text)

	rows_before_lang_filter = len(df)
	df = df[df.apply(is_english_row, axis=1)].copy()
	rows_after_lang_filter = len(df)

	rows_before_dedup = len(df)
	df = df.drop_duplicates(subset=TARGET_COLUMNS).reset_index(drop=True)
	rows_after_dedup = len(df)

	out_dir.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)

	print(f"Rows after English filter: {rows_after_lang_filter} (removed {rows_before_lang_filter - rows_after_lang_filter})")
	print(f"Rows after deduplication: {rows_after_dedup} (removed {rows_before_dedup - rows_after_dedup})")
	print(f"Rows after preprocessing: {rows_after_dedup} (removed total {rows_initial - rows_after_dedup})")
	print(f"Saved cleaned data to: {out_path}")


if __name__ == "__main__":
	main()
