# PDF Wizard Parser
**PDF Wizard Chunker** uses a heuristic method to parse the text from a pdf file that is 100x faster than a semantic chunker.
 This tool is useful for preprocessing large documents for text analysis, machine learning, or other data processing tasks.

### Features

- Extracts text from PDF files while ignoring page headers and footers.
- Splits the text into semantically meaningful chunks.
- Saves each chunk as an individual .txt file.
- Configurable chunk size via a command-line flag.
- Simple and intuitive command-line interface.

### Installation

Clone the repository:
```
git clone https://github.com/mouad157/pdf_wizard_parser/
cd pdf_wizard_parser
```

Install the required dependencies:

```
pip install -r requirements.txt
```
### Functions

#### ```semantic_chunker(text)```

Splits the input text into semantically meaningful chunks using a HuggingFace-based embedding model.

##### Parameters
- text (str): The text to be chunked.
##### Returns
- list: A list of semantically meaningful chunks of text.
##### Example
```
text = "This is a long piece of text. It will be split into meaningful chunks."
chunks = semantic_chunker(text)
print(chunks)
```

#### ```wizard_chunker(texts, chunk_size=300)```

Processes a given text into chunks based on sentence structure and a minimum chunk size.
##### Parameters
- texts (str): The input text to process.
- chunk_size (int, optional): The minimum number of tokens in each chunk. Default is 300.
##### Returns
- list: A list of text chunks.
##### Example
```
text = "This is the first sentence.\nThis is the second sentence.\nThis is the third sentence."
chunks = wizard_chunker(text, chunk_size=50)
print(chunks)
```

#### ```wizard_parser(file_path, question, chunk_size=300, k=5)```

Parses a PDF file to extract relevant chunks of text based on a user-provided question.

##### Parameters
- file_path (str): Path to the PDF file.
- question (str): The question or query to filter relevant chunks.
- chunk_size (int, optional): Minimum size of text chunks. Default is 300.
- k (int, optional): Number of top relevant chunks to return. Default is 5.
##### Returns
- list: A list of the top k relevant text chunks.
- int: Total number of chunks generated.
##### Example
```
file_path = "sample.pdf"
question = "What is the topic of the second section?"
relevant_chunks, total_chunks= wizard_parser(file_path, question)
print(f"Top chunks: {relevant_chunks}")
print(f"Total chunks generated: {total_chunks}")
```

### Usage


1. Run the script using the command-line interface to use the wizard pdf chunker:

```
python wizard_pdf_chunker.py -f path/to/your/file.pdf -o output_folder -c chunk_size
```

##### Arguments

- ```-f``` or ```--file``` (required): Path to the PDF file to process.
- ```-o``` or ```--output``` (optional): Directory where the text chunks will be saved. Defaults to chunks.
- ```-c``` or ```--chunk_size``` (optional): Minimum number of tokens per chunk. Defaults to 300.

##### Example

Process a file named ```sample.pdf``` with a chunk size of 300 tokens, saving the output in the ```output_chunks``` folder:

```python wizard_pdf_chunker.py -f sample.pdf -o output_chunks -c 300```

The script will create the folder ```output_chunks``` (if it doesn’t already exist) and save chunks as ```chunk_1.txt```, ```chunk_2.txt```, and so on.



2. Run the script using the command-line interface to use the semantic pdf chunker for comparison:

```
python semantic_pdf_chunker.py -f path/to/your/file.pdf -o output_folder -c chunk_size
```

##### Arguments

- ```-f``` or ```--file``` (required): Path to the PDF file to process.
- ```-o``` or ```--output``` (optional): Directory where the text chunks will be saved. Defaults to chunks.

##### Example

Process a file named ```sample.pdf``` , saving the output in the ```output_chunks``` folder:

```python semantic_pdf_chunker.py -f sample.pdf -o output_chunks -c 300```

The script will create the folder ```output_chunks``` (if it doesn’t already exist) and save chunks as ```chunk_1.txt```, ```chunk_2.txt```, and so on.

### Demo

to run the demo run the following code:

```
streamlit run demo.py
```
