# 加载 Embedding 模型，并提供统一的文本向量化接口 embed_documents() 和 embed_query() 
import dashscope
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModelManagerWithDashScope:
    """API 调用 DashScope 平台上的 Embedding 模型"""

    def __init__(
            self,
            model_name: str = "text-embedding-v4",
            device: str = "cpu",
            batch_size: int = 32,
        ) -> None:
        self.model_name = model_name

        # 初始化嵌入模型：
        self.embeddings = DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

class EmbeddingModelManagerWithHuggingFace:
    """本地 Embedding 模型的初始化， model from HuggingFace"""
    def __init__(
            self,
            model_name: str = "BAAI/bge-small-zh-v1.5",
            device: str = "cpu",
            batch_size: int = 32
        ) -> None:
        self.model_name = model_name

        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": device                # 本地运行，在 cpu 上
            },
            encode_kwargs={
                "batch_size": batch_size,       # 
                "normalize_embeddings": True    # 归一化向量，即向量除以自身模长，便于后续使用 FAISS 进行相似性检索
            }
        )
        # LangChain-huggingface 提供了统一的 Embedding 接口，主要包含两个方法：
        # embed_documents(texts)： 知识库中的多个文档转换为向量
        # embed_query(text)： 用户提出的问题转换为向量 

    # 我自己的模型再将 LangChain 提供的接口封装成自己的接口，提供给下游：
    def embed_documents(self, texts: list[str]):
        """
        多个文本转换为向量
        :return : 二维向量列表
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str):
        """
        单个查询文本转换为向量
        :return : 一维向量
        """
        return self.embeddings.embed_query(text)

    def get_embeddings(self):
        return self.embeddings