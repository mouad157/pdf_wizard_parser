from util import pdf_to_text, wizard_chunker
import argparse
import os

def pdf_to_chunks(pdf_file, output_folder, chunk_size=300):
    """
    Processes a PDF file into text chunks and saves them as individual .txt files.

    Args:
        pdf_file (str): Path to the PDF file.
        output_folder (str): Directory where chunks will be saved.
        chunk_size (int): Minimum number of tokens per chunk.
    """

    text,_ = pdf_to_text(pdf_file)
    chunks = wizard_chunker(text,chunk_size=chunk_size)
    # Save chunks to separate .txt files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i, chunk in enumerate(chunks):
        file_name = os.path.join(output_folder, f"chunk_{i+1}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(chunk)
    
    print(f"Saved {len(chunks)} chunks in the folder: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a PDF file into text chunks and save them as .txt files.")
    parser.add_argument("-f", "--file", required=True, help="Path to the PDF file")
    parser.add_argument("-o", "--output", default="chunks", help="Folder to save the text chunks")
    parser.add_argument("-c", "--chunk_size", type=int, default=300, help="Minimum tokens per chunk (default: 300)")
    
    args = parser.parse_args()
    pdf_to_chunks(args.file, args.output, args.chunk_size)