from sentence_transformers import SentenceTransformer, util
import torch
import argparse
import pdfplumber
from nltk.tokenize import word_tokenize
import re


def wizard_parser(file_path,question,chunk_size = 400, k=5):

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

    tex = ""
    for j in range(len(texts)-1):

        if texts[j] =="\n" and texts[j+1] == "\n" and len(word_tokenize(tex)) >= chunk_size:
            graph.append(tex)
            tex= ""
        else:
            tex = tex + texts[j]

    model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    query_embedding = model.encode(question)
    passage_embeddings = model.encode(graph)
    similarity = util.pytorch_cos_sim(query_embedding, passage_embeddings)
    top = torch.topk(similarity,5)
    indices = top.indices.squeeze().tolist()
    tert = [graph[i] for i in indices]
    return tert
