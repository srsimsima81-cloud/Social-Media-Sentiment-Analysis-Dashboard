import os

def create_folders():
    folders = ["models", "outputs", "images"]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)