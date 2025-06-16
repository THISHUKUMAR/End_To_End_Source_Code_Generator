from src.helper import repo_ingestion, load_repo, text_splitter, download_hugging_face_embeddings
from dotenv import load_dotenv
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_KEY')
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY



# url = "https://github.com/entbappy/End-to-end-Medical-Chatbot-Generative-AI"

# repo_ingestion(url)


documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = download_hugging_face_embeddings()



#storing vector in choramdb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()