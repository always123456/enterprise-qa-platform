from langchain_core.documents import Document

documents = [
    Document(
        page_content="人工智能是研究、开发用于模拟和扩展人类智能的理论、方法和技术。",
        metadata={
            "source": "ai_intro.pdf",
            "page": 1,
            "category": "人工智能"
        }
    ),
    Document(
        page_content="机器学习是人工智能的一个重要分支，主要研究如何让计算机从数据中学习规律。",
        metadata={
            "source": "machine_learning.pdf",
            "page": 2,
            "category": "机器学习"
        }
    ),
    Document(
        page_content="深度学习是机器学习的一个分支，主要使用多层神经网络进行特征学习。",
        metadata={
            "source": "deep_learning.pdf",
            "page": 3,
            "category": "深度学习"
        }
    )
]

from embedding_storage.embedding_model import EmbeddingModelManagerWithHuggingFace
from embedding_storage.faiss_manager import FAISSManager

embedding_model_manager = EmbeddingModelManagerWithHuggingFace(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cpu",
    batch_size=32
)

faiss_manager = FAISSManager(
    embedding_model_manager=embedding_model_manager,
    index_path="vector_db/faiss_index"
)

faiss_manager.build_index(documents)

# 保存索引
faiss_manager.save_index()