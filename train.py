from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from genre_model import GenreAutoencoder, GenreDataset
import sys

DATA_DIR = Path("./data")
CHECKPOINT_FILE = DATA_DIR / "genre_opposition_model.pt"
GENRE_MATRIX_FILE = DATA_DIR / "genre_matrix.npy"
GENRES_FILE = DATA_DIR / "genres.npy"



def load_data():
    movies = pd.read_csv(DATA_DIR / 'movies.csv')

    # Filter out some films

    # Filter out films older than 1980 (arbitrary)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)", expand=False).astype(float)
    movies = movies[pd.isna(movies["year"]) | (movies["year"] >= 1980)]
    # Remove films with no genre
    movies = movies[movies['genres'] != "(no genres listed)"]

    movies['genre_list'] = movies['genres'].str.split("|")
    # Remove documentaries (in the embedding space, documentaries are so unlike everything else that they always are the least similar)
    movies = movies[movies['genre_list'].apply(lambda g: 'Documentary' not in g)].reset_index(drop=True) 

    return movies

def build_genre_matrix(movies, force_rebuild=False):
    if GENRE_MATRIX_FILE.exists() and GENRES_FILE.exists() and not force_rebuild:
        print("Loading genre matrix...")
        X = np.load(GENRE_MATRIX_FILE)
        genres = np.load(GENRES_FILE)
    else:
        print("Building genre matrix...")
        mlb = MultiLabelBinarizer(sparse_output=False)
        genre_matrix = mlb.fit_transform(movies['genre_list'])
        genres = mlb.classes_.tolist()
        G = genre_matrix.shape[1]


        print(f"{len(movies)} movies, {G} genres: {genres}")

        # Weight genres by how commmon they are

        def idf_weight(matrix):
            n = matrix.shape[0]
            doc_counts = matrix.sum(axis=0)
            idf = np.log((1 + n) / (1 + doc_counts)) + 1.0
            return matrix * idf[np.newaxis, :]

        X = idf_weight(genre_matrix)
        X = X / X.max()

        np.save(GENRE_MATRIX_FILE, X)
        np.save(GENRES_FILE, genres)

    return X, genres

def preprocess():
    movies = load_data()
    X, genres = build_genre_matrix(movies)
    return X, genres

def train(EPOCHS = 100, resume=True):
    X, genres = preprocess()
    ds = GenreDataset(X)

    model = GenreAutoencoder(len(genres))

    loader = DataLoader(ds, batch_size=512, shuffle=True) 
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.1)

    loss_fn = nn.MSELoss()

    start_epoch = 0

    if resume and CHECKPOINT_FILE.exists():
        print(f"Loading checkpoints: {CHECKPOINT_FILE}")
        
        checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])

        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']+1

        print(f"Resuming from epoch {start_epoch}")
    
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(start_epoch, start_epoch+EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in loader:
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * batch.size(0)
        scheduler.step()
        avg = total_loss / len(ds)

        print(f"Epoch { epoch} loss={avg}")

    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss" : avg,
        },
        CHECKPOINT_FILE
    )

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    train(epochs)
