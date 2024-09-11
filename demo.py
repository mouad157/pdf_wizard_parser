import streamlit as st
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from streamlit_pdf_viewer import pdf_viewer
import streamlit_scrollable_textbox as stx
import torch
import pdfplumber
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stpwrd = stopwords.words('english')
import re

def parse(pdfu,chunk_size):
    paginas = []
    with pdfplumber.open(pdfu) as pdf:
        total = ""

        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            pagei = page.within_bbox((0, page.height*0.05 , page.width, page.height-page.height*0.05))
            text = pagei.extract_text(layout=True)
            paginas.append(text)
            total = total + text

    graph = []
    texts1 = re.sub(r'\n\s+\n\s+\n', '\n\n\n', total)
    texts = re.sub(r' +', ' ', texts1)
    # tex = ""
    graph3= []
    original = ""
    sentences = texts.split("\n")
    for i in range(len(sentences)):
        p=True
        tokenized = word_tokenize(sentences[i])
        if len(tokenized) == 0:
            continue
        words = [word for word in tokenized if word not in stpwrd]
        #print(words)
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
            original = original +"\n"+ sentences[i]+"\n"
            
        else:
            original = original + sentences[i]
        if i == len(sentences) -1:
            graph3.append(original)

    return graph3


def search(model, question, texts):
    query_embedding = model.encode(question)
    passage_embeddings = model.encode(texts)
    similarity = util.pytorch_cos_sim(query_embedding, passage_embeddings)
    top = torch.topk(similarity,3)
    indices = top.indices.squeeze().tolist()
    out = [texts[i] for i in indices]
    return indices, out


def main():
    st.set_page_config(
    page_title="Wizard parser",
    page_icon="🧊",
    layout="wide")
    JOB_HTML_TEMPLATE = """
    <div style='color:#444'>
    <h4>{}</h4>
    <h6>{}</h6>
    </div>
    """
    st.title("PDF Wizard Parser")

    with st.spinner('Loading the model. Please wait.'):
        model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    st.success("Loaded the model!")
    
    # Nav  Search Form
    with st.form(key='searchform'):
        nav1,nav2,nav3,nav4 = st.columns([5,3,2,2])
        with nav1:
            search_term = st.text_input("What is your question")
        with nav2:
            file =st.file_uploader("Upload a PDF file", type="pdf")
        with nav3:
            chunk_size = st.text_input("Chunk_size")
        with nav4:
            st.text("Search")
            submit_search = st.form_submit_button(label='Search')
    
    if submit_search:
        with st.spinner('Parsing the PDF file. Please wait.'):
            texts = parse(file,int(chunk_size))
        st.success("pdf is parsed!")
        st.success("You searched for: {}".format(search_term))
        content,mtd = search(model,search_term,texts)
        num_of_results = len(mtd)
        st.subheader("Showing {} results".format(num_of_results))
        binary_data = file.getvalue()
        pdf_viewer(input=binary_data,
                width=700,height=600)
        for i in mtd:
            body = i
            with st.expander(body.split('\n')[1]):
                stx.scrollableTextbox(body[1:],height = 300)
if __name__ == '__main__':
	main()