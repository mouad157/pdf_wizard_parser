import argparse
import re
import time
import nltk
import pdfplumber
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')
stpwrd = stopwords.words('english')

def pdf_to_text(file_path):
    """
    Extracts text from a PDF file, cleaning and returning both the combined text and individual pages.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        tuple: Cleaned full text and a list of text from individual pages.
    """
    with pdfplumber.open(file_path) as pdf:
        total_text = ""
        pages_text = []

        for page in pdf.pages:
            # Extract text within a specific bounding box
            page_text = page.within_bbox((0, page.height * 0.18, page.width, page.height - page.height * 0.18)).extract_text(layout=True) or ""
            pages_text.append(page_text)
            total_text += page_text

    # Clean text by normalizing spaces and newlines
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', re.sub(r' +', ' ', total_text))
    return cleaned_text, pages_text

def semantic_chunker(text):
    """
    Splits the text into semantically meaningful chunks using a HuggingFace-based chunker.
    
    Args:
        text (str): The text to be chunked.
    
    Returns:
        list: A list of chunked documents.
    """
    embeddings = HuggingFaceEmbeddings()
    text_splitter = SemanticChunker(embeddings)
    docs = text_splitter.create_documents([text])
    return [doc.page_content for doc in docs]

def wizard_chunker(texts, chunk_size=300):
    """
    Processes a given text into chunks based on sentence content and a minimum chunk size.

    Args:
        texts (str): The input text to process.
        chunk_size (int): Minimum number of tokens in a chunk.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    current_chunk = []
    sentences = texts.split("\n")
    
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if not tokens:
            current_chunk.append(sentence)
            continue

        # Filter out stopwords and check if words qualify
        filtered_words = [word for word in tokens if word not in stpwrd]
        if len(filtered_words) > 1 and all(word[0].isupper() or word[0].isdigit() for word in filtered_words):
            # Add sentence to the current chunk
            if sum(len(word_tokenize(sent)) for sent in current_chunk) >= chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(sentence)
            
        else:
            current_chunk.append(sentence)
    
    # Append any remaining chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks

def wizard_parser(file_path,question,chunk_size = 300, k=5):

    """
    Parses a PDF file to extract relevant chunks of text based on a given question.
    
    Args:
        file_path (str): Path to the PDF file.
        question (str): The question to use for filtering relevant chunks.
        chunk_size (int): Minimum size of text chunks.
        k (int): Number of relevant chunks to return.

    Returns:
        list: Top k relevant text chunks.
        int: Total number of chunks.
        float: Duration of processing time.
    """
    # Load and extract text from the PDF
    with pdfplumber.open(file_path) as pdf:
        total_text = ''.join(
            page.within_bbox((0, page.height * 0.18, page.width, page.height - page.height * 0.18)).extract_text(layout=True) or ""
            for page in pdf.pages
        )
    
    # Clean and process text
    cleaned_text = re.sub(r'\n\s+\n', '\n\n', re.sub(r' +', ' ', total_text))
    sentences = cleaned_text.split("\n")
    stop_words = set(stopwords.words('english'))

    # Chunk processing
    chunks = []
    current_chunk = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Check if the sentence qualifies as part of a chunk
        if len(filtered_words) > 1 and all(word[0].isupper() or word[0].isdigit() for word in filtered_words):
            current_chunk.append(sentence)
            if sum(len(word_tokenize(sent)) for sent in current_chunk) >= chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        else:
            current_chunk.append(sentence)
    
    # Append any remaining chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    
    # Calculate chunk embeddings and similarity
    model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    query_embedding = model.encode(question)
    chunk_embeddings = model.encode(chunks)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings).squeeze()
    
    # Get top-k relevant chunks
    top_indices = torch.topk(similarities, k).indices.tolist()
    top_chunks = [chunks[i] for i in top_indices]

    return top_chunks, len(chunks)

if __name__ == '__main__':
    text,_ = pdf_to_text("./wdg.pdf")
    start_time = time.time()
    chunks = semantic_chunker(text)
    duration = time.time() - start_time
    print(duration)
    start_time2 = time.time()
    chunks2 = wizard_chunker(text)
    duration2 = time.time() - start_time2
    print(duration2)
    print(len(chunks),len(chunks2))
    print(wizard_parser("./wdg.pdf", "how to setup an access point")[1])
