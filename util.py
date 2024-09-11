from sentence_transformers import SentenceTransformer, util
import torch
import argparse
import pdfplumber
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
nltk.download('stopwords')
stpwrd = stopwords.words('english')

def wizard_parser(file_path,question,chunk_size = 300, k=5):

    #This function takes a pdf file path, a question, chunk size and a integer k and returns a list of k useful chunks from 
    # the pdf file that are at least in the size of chunk_size

    paginas = []
    with pdfplumber.open(file_path) as pdf:
        total = ""

        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            pagei = page.within_bbox((0, page.height*0.18 , page.width, page.height-page.height*0.18))
            text = pagei.extract_text(layout=True)
            paginas.append(text)
            total = total + text

    graph = []
    texts1 = re.sub(r'\n\s+\n\s+\n', '\n\n\n', total)
    texts = re.sub(r' +', ' ', texts1)
    graph2 = []
    graph3= []
    clean = ""
    original = ""
    sentences = texts.split("\n")
    for i in range(len(sentences)):
        p=True
        tokenized = word_tokenize(sentences[i])
        if len(tokenized) == 0:
            continue
        words = [word for word in tokenized if word not in stpwrd]
        print(words)
        if len(words) <= 1:
            p= False
        for l in words:
            if l[0].isupper() or l[0].isdigit():
                continue
            else:
                p=False
                break
        if p:
            tokens = word_tokenize(original)
            if len(tokens) >= chunk_size:
                graph3.append(original)
                original = ""
            original =  original +"\n"+ sentences[i]+"\n"
            
        else:
            original = original + sentences[i]
        if i == len(sentences) -1:
            graph3.append(original)

    model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    query_embedding = model.encode(question)
    passage_embeddings = model.encode(graph3)
    similarity = util.pytorch_cos_sim(query_embedding, passage_embeddings)
    top = torch.topk(similarity,3)
    indices = top.indices.squeeze().tolist()
    tert = [graph3[i] for i in indices]
    return tert,len(graph3)
if __name__ == '__main__':
	print(wizard_parser("./wdg.pdf", "how to setup an access point"))