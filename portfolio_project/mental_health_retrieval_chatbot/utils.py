from datasets import load_dataset
import os
import tarfile
import requests
import pandas as pd


def download_and_load_empathetic_dialogues(data_dir="resources"):
    url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    tar_path = os.path.join(data_dir, "empatheticdialogues.tar.gz")

    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download the file if not already present
    if not os.path.exists(tar_path):
        print("Downloading EmpatheticDialogues dataset...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("File already downloaded.")

    # Extract the tar.gz
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
        print("Extraction complete.")

    # Load the train set into a pandas DataFrame
    train_path = os.path.join(data_dir, "empatheticdialogues", "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        print("Loaded training data. Rows:", len(df))
    else:
        raise FileNotFoundError(f"Expected file not found at: {train_path}")

    return df


def download_and_load_counsel_chat():
    # Load the Counsel Chat dataset from Hugging Face
    print("Loading Counsel Chat dataset...")
    dataset = load_dataset("nbertagnolli/counsel-chat")
    print("Dataset loaded successfully.")

    # Convert to pandas DataFrame
    df = dataset["train"].to_pandas()
    print("Converted to DataFrame. Rows:", len(df))

    return df
