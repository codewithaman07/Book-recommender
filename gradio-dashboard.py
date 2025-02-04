import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests

from langchain_community.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

# Load the books data
try:
    books = pd.read_csv("books_with_emotions.csv")
except FileNotFoundError:
    print("Error: The books data file was not found.")
    # Handle the error or show a message on the dashboard

# Check if the image URL is valid
def check_image(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except:
        return False

# Update large_thumbnail with a more robust check
books["large_thumbnail"] = books["thumbnail"].apply(
    lambda x: x + "&fife=w800" if pd.notna(x) and x != "" else "cover-not-found.jpg"
)

# Ensure thumbnails are valid
books["large_thumbnail"] = books["large_thumbnail"].apply(
    lambda x: x if check_image(x) else "cover-not-found.jpg"
)

# Load the raw documents (book descriptions)
try:
    raw_documents = TextLoader("tagged_descriptions.txt", encoding="utf-8").load()
except FileNotFoundError:
    print("Error: The tagged descriptions file was not found.")
    # Handle the error or show a message on the dashboard

# Split the documents into chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Initialize HuggingFace embeddings (replace OpenAI embeddings)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the Chroma vector store
db_books = Chroma.from_documents(documents, embedding_model)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    # Perform similarity search
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Retrieve books by ISBN
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)]

    # Filter by category if necessary
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort by tone if specified
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    # If no recommendations are found, display a placeholder
    if recommendations.empty:
        return [("no_results.jpg", "No recommendations found")]

    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)


if __name__ == "__main__":
    dashboard.launch()
