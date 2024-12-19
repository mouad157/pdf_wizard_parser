# PDF Wizard Parser

the PDF Wizard Parser uses a heuristic method to parse the text from a pdf file that is 100x faster than a semantic chunker.

to use the pdf parser you can run the following code:

```
from util import wizard_parser
list_of_context = wizzard_parser("example.pdf","Where is Singapore?", chunksize = 400, k = 5)
```
### Usage

1 - Run the script using the command-line interface to use the wizard pdf chunker:

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



2 - Run the script using the command-line interface to use the semantic pdf chunker for comparison:

```
python semantic_pdf_chunker.py -f path/to/your/file.pdf -o output_folder -c chunk_size
```

##### Arguments

- ```-f``` or ```--file``` (required): Path to the PDF file to process.
- ```-o``` or ```--output``` (optional): Directory where the text chunks will be saved. Defaults to chunks.

##### Example

Process a file named ```sample.pdf``` with a chunk size of 300 tokens, saving the output in the ```output_chunks``` folder:

```python semantic_pdf_chunker.py -f sample.pdf -o output_chunks -c 300```

The script will create the folder ```output_chunks``` (if it doesn’t already exist) and save chunks as ```chunk_1.txt```, ```chunk_2.txt```, and so on.





to run the demo run the following code:


```
streamlit run demo.py
```
