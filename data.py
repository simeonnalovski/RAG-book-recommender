import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import ast

books_path = 'goodreads_data.csv'
index_path = 'book_index.faiss'
metadata_path = 'book_metadata.csv'

df = pd.read_csv(books_path)
df['Genres'] = df['Genres'].fillna('[]')

def parse_genres(cell):
    if isinstance(cell, str):
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

        return [cell]
    return cell

df['Genres'] = df['Genres'].apply(parse_genres)
df['Genres_str'] = df['Genres'].apply(lambda genres: ', '.join(genres))


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df['Genres_str'].tolist(), show_progress_bar=True)

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, index_path)
df[['Book', 'Genres']].to_csv(metadata_path, index=False)
