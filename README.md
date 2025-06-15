# RAG-book-recommender
This project is a content-based book recommendation system that leverages FAISS for similarity search, Sentence Transformers for generating embeddings from book metadata, and a lightweight language model (Qwen 0.6B) to provide human-like explanations for the recommendations. The user interface is built using Streamlit for an interactive experience.

Features:

    Search for a book by title (supports partial input)

    Receive top-N similar book recommendations based on genres

    FAISS-powered vector search for fast semantic similarity

    LLM-generated explanations for why each recommendation makes sense

    Uses book metadata including genres, average rating, and number of ratings

Tech Stack:

    Python

    Streamlit – Web UI

    FAISS – Approximate Nearest Neighbor (ANN) search

    SentenceTransformer (all-MiniLM-L6-v2) – Embedding genres

    Qwen 0.6B via Hugging Face – Language model for natural language generation

    Pandas / NumPy – Data handling and vector operations

Project Structure:

project_root/

    app.py → Main Streamlit application

    book_index.faiss → FAISS vector index of genre embeddings

    book_metadata.csv → Dataset with book titles, genres, and ratings

    README.txt → Project documentation

How It Works:

    User enters a book title or part of one.

    The system finds matching titles from the dataset.

    The genre of the selected book is embedded using SentenceTransformer.

    FAISS index is queried to find books with similar genre embeddings.

    Recommendations are shown alongside LLM-generated explanations.

Knowledge-Based System Design:

The application uses a content-based knowledge-based system, where knowledge is encoded in the form of semantic vectors and metadata. The FAISS index acts as a knowledge base, performing efficient similarity search. The system does not rely on user data, making it ideal for domain-based recommendation.

Setup Instructions:

    Install Python dependencies manually:

    pip install streamlit faiss-cpu pandas numpy sentence-transformers transformers

    Run the app:

    streamlit run app.py

Make sure the files book_metadata.csv and book_index.faiss are in the same directory as app.py.
NOTE: replace YOUR_HF_TOKEN with the actual value of your HF toketn
