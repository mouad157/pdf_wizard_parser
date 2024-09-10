import streamlit as st
import streamlit.components.v1 as stc
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from streamlit_pdf_viewer import pdf_viewer
import streamlit_scrollable_textbox as stx
import torch
import pdfplumber
import re

def parse(pdfu):
    paginas = []
    with pdfplumber.open(pdfu) as pdf:
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

        if texts[j] =="\n" and texts[j+1] == "\n" and len(word_tokenize(tex)) >= 400:
            graph.append(tex)
            tex= ""
        else:
            tex = tex + texts[j]
    return graph


def search(model, question, texts):
    query_embedding = model.encode(question)
    passage_embeddings = model.encode(texts)
    similarity = util.pytorch_cos_sim(query_embedding, passage_embeddings)
    top = torch.topk(similarity,5)
    indices = top.indices.squeeze().tolist()
    out = [texts[i] for i in indices]
    # for k in out:
    #     terc = serach
    return indices, out


def main():
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
        nav1,nav2,nav3 = st.columns([5,3,2])

        with nav1:
            search_term = st.text_input("What is your question")
        with nav2:
            file =st.file_uploader("Upload a PDF file", type="pdf")
            
        with nav3:
            st.text("Search")
            submit_search = st.form_submit_button(label='Search')
    
    if submit_search:
        with st.spinner('Parsing the PDF file. Please wait.'):
            texts = parse(file)
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
            # st.markdown(JOB_HTML_TEMPLATE.format(body.split('\n')[0],body),
            # 		unsafe_allow_html=True)
            with st.expander(body.split('\n')[2]):
                stx.scrollableTextbox(body[2:],height = 300)
if __name__ == '__main__':
	main()