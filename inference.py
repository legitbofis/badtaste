from pathlib import Path
import numpy as np
import torch
#from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("./ml-32m")
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"


def get_embeddings(model, X, force_rebuild=False, device=None):
    if EMBEDDINGS_FILE.exists() and not force_rebuild:
        print("Loading embeddings...")
        Z_norm = np.load(EMBEDDINGS_FILE)
    else:
        print("Computing embeddings...")
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            _, Z = model(X_t)
            Z = np.array(Z.cpu().tolist())

        # normalise embedding vecotrs
        Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

        print(f"Saving embeddings to {EMBEDDINGS_FILE}")
        np.save(EMBEDDINGS_FILE, Z_norm)
    return Z_norm

def inference(model, X, movies_df, movie_ids):
    Z_norm = get_embeddings(model, X)

    indices = [
        movies_df.index[movies_df["movieId"] == mid][0]
        for mid in movie_ids
    ]

    query_embedding = Z_norm[indices].mean(axis=0, keepdims=True)

    # Conceptually, we are using cosine similarity. However, to make it faster,
    # we normalised the embeddings when we first built them and so now we can
    # simply take the dot product between two vectors. This is all mathematically
    # identical to cosine similarity.

    #sims = cosine_similarity(query_embedding, Z_norm)[0]
    sims = np.dot(query_embedding, Z_norm.T)[0]

    for idx in indices:
        sims[idx] = 1.0

    opposite_idx = np.argsort(sims)[:5]

    return movies_df.iloc[opposite_idx]
