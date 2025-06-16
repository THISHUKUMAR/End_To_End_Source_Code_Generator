import os
from git import Repo
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.generic import GenericLoader
# from langchain.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



#clone any github repositories 
# def repo_ingestion(repo_url):
#     os.makedirs("repo", exist_ok=True)
#     repo_path = "repo/"
#     Repo.clone_from(repo_url, to_path=repo_path)


import shutil
import stat


def handle_remove_readonly(func, path, _):
    """Clear the read-only bit and reattempt the removal."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def repo_ingestion(repo_url):
    repo_path = "repo/"

    # Delete the existing folder if it exists (handling Windows permission issues)
    if os.path.exists(repo_path):
        print(f"[INFO] Removing existing directory: {repo_path}")
        shutil.rmtree(repo_path, onerror=handle_remove_readonly)

    # Now safely clone
    print(f"[INFO] Cloning repo from {repo_url}")
    Repo.clone_from(repo_url, to_path=repo_path)


#Loading repositories as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language=Language.PYTHON, parser_threshold=500)
                                        )
    
    documents = loader.load()

    return documents




#Creating text chunks 
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                             chunk_size = 2000,
                                                             chunk_overlap = 200)
    
    text_chunks = documents_splitter.split_documents(documents)

    return text_chunks




def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings