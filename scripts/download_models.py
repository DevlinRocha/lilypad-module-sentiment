import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def download_models():
    MODEL_IDENTIFIER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    try:
        print(f"Downloading model `{MODEL_IDENTIFIER}`...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_IDENTIFIER)
        tokenizer.save_pretrained("./models")
        model.save_pretrained("./models")
        print("✅ Models downloaded successfully.")
    except Exception as error:
        print(
            f"❌ Error downloading models: {error}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    download_models()
