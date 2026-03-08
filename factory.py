from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_conf


class ChatModelFactory:
    def generate(self):
        return ChatTongyi(model=rag_conf['chat_model_name'])


class EmbeddingModelFactory:
    def generate(self):
        return DashScopeEmbeddings(model=rag_conf['embedding_model_name'])
