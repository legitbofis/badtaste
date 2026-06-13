import sys
from pathlib import Path
import torch
import pandas as pd
from inference import inference
from train import load_data
from train import build_genre_matrix
from genre_model import GenreAutoencoder

DATA_DIR = Path("./ml-32m")
MODEL_FILE = DATA_DIR / "genre_opposition_model.pt"

def load_model(num_genres, map_location='cpu'):
    device = torch.device(map_location)
    model = GenreAutoencoder(num_genres,).to(device)
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    # checkpoint may contain state_dict or the model directly
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def recommend(ids):
    movies = load_data()

    X, genres = build_genre_matrix(movies)

    model = load_model(len(genres), map_location='cpu')

    results = inference(model, X, movies, ids)


    # Get TMDB movie ids
    links = pd.read_csv(DATA_DIR / 'links.csv')

    results_with_tmdb = results.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    tmdb_ids = results_with_tmdb['tmdbId'].astype(int).tolist()

    print(tmdb_ids)
    
    
    return tmdb_ids

def _parse_args_as_ints(argv):
    ints = []
    for a in argv:
        try:
            ints.append(int(a))
        except ValueError:
            raise ValueError(f"Argument not an integer: {a!r}")
    return ints

if __name__ == '__main__':
    try:
        args = sys.argv[1:]
        if not args:
            raise SystemExit("Usage: python recommend.py <movieId> <movieId> ...")
        ids  = _parse_args_as_ints(args)
        recommend(ids)
    except ValueError as e:
        print("Error:", e)
        sys.exit(1)


