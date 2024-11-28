import os
import json
import numpy as np
import tkinter as tk
from stopwordsiso import stopwords
from scipy.sparse import save_npz, load_npz
from underthesea import word_tokenize, text_normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Vietnamese stop words
stop_words = stopwords("vi")

def preprocess_text(text):
    """
    Perform text normalization, word tokenization, and stop word removal.
    """
    text = text_normalize(text)
    word_list = word_tokenize(text, format="text").split()
    filtered_words = [word for word in word_list if word not in stop_words]
    return ' '.join(filtered_words).lower()


def load_text_data(directory):
    """
    Load text data from JSON files.
    Return list of documents (str).
    """
    text_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                string_list = [data["Title"], data["Detail_sapo"]] +  data["Content"]
                text = ' '.join(string_list)
                text_data[filename] = preprocess_text(text)
    return text_data


def save_vectorizer_data(save_path, vectorizer, tfidf_matrix, doc_names):
    vectorizer_data = {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist()
    }

    save_npz(os.path.join(save_path, 'tfidf_matrix.npz'), tfidf_matrix)
    with open(os.path.join(save_path, 'vectorizer.json'), 'w') as fout:
        json.dump(vectorizer_data, fout)
    with open(os.path.join(save_path, 'doc_names.json'), 'w') as fout:
        json.dump(doc_names, fout)


def load_vectorizer_data(save_path):
    """
    Return vectorizer, tfidf_matrix, doc_names.
    """
    tfidf_matrix = load_npz(os.path.join(save_path, 'tfidf_matrix.npz'))
    with open(os.path.join(save_path, 'vectorizer.json'), 'r') as fin:
        vectorizer_data = json.load(fin)
    with open(os.path.join(save_path, 'doc_names.json'), 'r') as fin:
        doc_names = json.load(fin)
        
    vectorizer = TfidfVectorizer()
    vectorizer.vocabulary_ = vectorizer_data["vocabulary"]
    vectorizer.idf_ = np.array(vectorizer_data["idf"])

    return vectorizer, tfidf_matrix, doc_names


def vectorize_text_data(text_data):
    """
    Vectorize the text data using TfidfVectorizer,
    or load vectorized data from files if saved data exists.
    """
    save_path = "./vectorizer_data"
    vectorizer, tfidf_matrix, doc_names = None, None, None
    if os.path.exists(save_path):
        vectorizer, tfidf_matrix, doc_names = load_vectorizer_data(save_path)
    else:
        vectorizer = TfidfVectorizer()
        doc_names = list(text_data.keys())
        tfidf_matrix = vectorizer.fit_transform(text_data.values())
        os.mkdir(save_path)
        save_vectorizer_data(save_path, vectorizer, tfidf_matrix, doc_names)

    return vectorizer, tfidf_matrix, doc_names


def search(query, vectorizer, tfidf_matrix, doc_names):
    """
    Search for query terms and rank by cosine similarity.
    """
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(tfidf_matrix, query_vector).flatten()

    ranked_results = sorted(
        [(doc_names[i], scores[i]) for i in range(len(doc_names))], 
        key=lambda x: x[1], 
        reverse=True
    )
    return [result for result in ranked_results if result[1] > 0]


def run_gui(vectorizer, tfidf_matrix, doc_names):
    """
    GUI for search input and results display
    """
    def on_search():
        query = entry.get()
        if not query:
            tk.messagebox.showwarning("Input Error", "Please enter a search query.")
            return
        
        top_k = int(top_k_entry.get()) if top_k_entry.get().isdigit() else 10
        results = search(query, vectorizer, tfidf_matrix, doc_names)
        
        if results:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Top search results:\n")
            for doc_id, score in results[:top_k]:
                result_text.insert(tk.END, f"{doc_id}: Cosine similarity score = {score:.4f}\n")
        else:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "No results found.")
    
    root = tk.Tk()
    root.title("Text Search Engine")

    tk.Label(root, text="Enter your search query:").pack(pady=5)
    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)

    tk.Label(root, text="Top K results:").pack(pady=5)
    top_k_entry = tk.Entry(root, width=5)
    top_k_entry.pack(pady=5)
    top_k_entry.insert(0, "10")

    tk.Button(root, text="Search", command=on_search).pack(pady=10)

    result_text = tk.Text(root, height=15, width=80)
    result_text.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    data_dir = "./data"
    text_data = load_text_data(data_dir)
    vectorizer, tfidf_matrix, doc_names = vectorize_text_data(text_data)

    run_gui(vectorizer, tfidf_matrix, doc_names)
