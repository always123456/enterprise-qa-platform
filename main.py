# 集成测试：
from unittest import TestCase
from langchain_core.documents import Document
from data_preprocess.data_load import DataLoaderModule
from embedding_storage.faiss_manager import FAISSManager
from embedding_storage.embedding_model import EmbeddingModelManagerWithHuggingFace
from prompt_engineering.prompt_manager import PromptManager
from common.model import DeepSeekChatModel
from task_chain.task_chain_module import TaskChainModule
from task_chain.debug_callback import DebugCallbackHandler
from agent_system.enterprise_agent_system import EnterpriseAgentSystem
from agent_system.document_storage import DocumentStorage

# 集成测试类：
class TestEnterpriseQASystem(TestCase):
    def setUp(self):
        # 设置数据加载和向量存储路径：
        self.document_directory = "dataset/"

        # 初始化各个模块：
        print("初始化数据加载模块...")
        self.data_loader = DataLoaderModule()
        self.loaded_docs: list[Document] = self.data_loader.load_batch(directory=self.document_directory)

        print("初始化嵌入生成与存储模块...")
        self.embedding_storage = FAISSManager(embedding_model_manager=EmbeddingModelManagerWithHuggingFace())
        self.faiss = self.embedding_storage.build_index(documents=self.loaded_docs)

        print("初始化提示词工程模块...")
        self.prompt_engineering = PromptManager(model=DeepSeekChatModel())

        print("初始化任务链模块和回调处理模块...")
        self.task_chain = TaskChainModule(retriever=self.embedding_storage.as_retriever())
        self.callback_handler = DebugCallbackHandler()

        print("初始化企业级 Agent 系统...")
        self.enterprise_agent = EnterpriseAgentSystem(model=DeepSeekChatModel(),document_storage=DocumentStorage())

    def test_complete_qa_flow(self):
        """
        集成测试完整的问答流程，从文档加载、嵌入生成到任务链执行和Agent选择
        """

        query = "公司的年度增长策略是什么？"

        