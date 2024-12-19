import os
import argparse
from util import pdf_to_text, semantic_chunker

def pdf_to_chunks(pdf_file, output_folder):
    """
    Processes a PDF file into text chunks using a semantic chunker and saves them as individual .txt files.

    Args:
        pdf_file (str): Path to the PDF file.
        output_folder (str): Directory where chunks will be saved.
    """
    text = pdf_to_text(pdf_file)
    chunks = semantic_chunker(text)

    # Save chunks to separate .txt files
    for i, chunk in enumerate(chunks):
        file_name = os.path.join(output_folder, f"chunk_{i+1}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(chunk)
    
    print(f"Saved {len(chunks)} chunks in the folder: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a PDF file into text chunks and save them as .txt files.")
    parser.add_argument("-f", "--file", required=True, help="Path to the PDF file")
    parser.add_argument("-o", "--output", default="chunks", help="Folder to save the text chunks")
    
    args = parser.parse_args()
    pdf_to_chunks(args.file, args.output)