import os 
import sys
import re
from typing import Callable
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
sys.path.append("./")
from retriever.embedding import EmbeddingRetriever
from utils.device import get_device
from utils.messages import MessageHistory
from models.abstract import AbstractModel

DIR = "retriever"
class HyDERetriever(EmbeddingRetriever):

    """
    A retriever which utilizes the Hypothetical Document Embedding approach
    """


    def __init__(self, hyde_model : AbstractModel, docs_path : str = "papers", vectors_path : str = "vectorstore", recreate : bool = False):
        super().__init__(docs_path, vectors_path, recreate)
        self.hyde_model = hyde_model
        self.messages = MessageHistory("Your job is to write excerpts from biomedical research papers that contain the answer to a user's question. These excerpts are from papers that are not real, your job is to write what you think the ansert is as if it were from a biomedical, health research, paper.")
        self.messages.add_user_message("What is Streptococcus agalactiae a leading cause of in the United States?")
        self.messages.add_model_message("Streptococcus agalactiae (group B streptococcus [GBS]) infection in pregnant women is the leading cause of infectious neonatal morbidity and mortality in the United States.")
        self.messages.add_user_message("What activities were conducted as part of the research efforts?")
        self.messages.add_model_message("Results: Activities were conducted such as handing out materials, providing educational resources, contacting participants, and stakeholders, as well as coordinating collaboration with community and organizations.")

    def __call__(self, query : str, n : int) -> list[Document]:
        messages = self.messages.copy()
        messages.add_user_message(query)
        hyde_doc = self.hyde_model(messages)
        return super().__call__(query + "\n" + hyde_doc, n)