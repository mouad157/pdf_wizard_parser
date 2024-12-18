# PDF Wizard Parser

the PDF Wizard Parser uses a heuristic method to parse the text from a pdf file that is 100x faster than a semantic chunker.

to use the pdf parser you can run the following code:

```
from util import wizard_parser
list_of_context = wizzard_parser("example.pdf","Where is Singapore?", chunksize = 400, k = 5)
```


to run the demo run the following code:


```
streamlit run demo.py
```
