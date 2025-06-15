
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st


index = faiss.read_index("book_index.faiss")
metadata = pd.read_csv("book_metadata.csv")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
hf_token = "YOUR_HF_TOKEN"
llm = pipeline("text-generation", model="Qwen/Qwen3-0.6B-Base", token=hf_token, device_map=None,trust_remote_code=True)

def get_matching_books(search_term):
    return metadata[metadata['Book'].str.contains(search_term, case=False, na=False)]['Book'].unique().tolist()

def recommend_books(input_title, top_k=6):
    input_row = metadata[metadata['Book'].str.lower() == input_title.lower()]
    if input_row.empty:
        return f"Book titled '{input_title}' not found in the dataset."

    input_genres = input_row.iloc[0]['Genres']
    input_embedding = embedding_model.encode([input_genres])
    distances, indices = index.search(np.array(input_embedding), top_k)

    recommended = metadata.iloc[indices[0]]
    recommended_books = recommended[recommended['Book'].str.lower() != input_title.lower()]

    prompt = f"""The user liked the book titled "{input_title}"\n
Here are 5 similar books, which the user should read after "{input_title}":\n"""

    for _, row in recommended_books.iterrows():
        prompt += f"{row['Book']}\n"

    prompt += ("\nExplain in 1-2 sentences per book why each of the following books is a good recommendation:\n"
               "Also, end each recommendation end with a new line")

    for _, row in recommended_books.iterrows():
        prompt += f"'{row['Book']}': "

    result = llm(prompt, max_new_tokens=512, do_sample=True)[0]['generated_text']
    return result

st.title(" Book Recommender")

user_input = st.text_input("Enter a book title (partial or full):")

if user_input:
    matching_books = get_matching_books(user_input)
    if matching_books:
        selected_book = st.selectbox("Select a book from matches:", matching_books)
        if st.button("Recommend"):
            with st.spinner("Generating recommendations..."):
                output = recommend_books(selected_book)
                st.text_area(" Recommendations:", output, height=400)
    else:
        st.warning("No matching books found. Please try another title.")
