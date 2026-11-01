if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    from agent_system.document_storage import DocumentStorage

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    storage = DocumentStorage(
        embeddings=embeddings,
        index_path="faiss_index"
    )

    # 加载企业文档
    documents = storage.load_documents(
        "dataset/"
    )

    # 构建向量库
    storage.build_vector_store(documents)

    # 持久化
    storage.save()

    # 检索测试
    results = storage.similarity_search(
        "员工如何申请年假？",
        k=4
    )

    for doc in results:
        print(doc.metadata)
        print(doc.page_content)