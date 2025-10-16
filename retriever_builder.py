from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from configs import RetrieverConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from concurrent.futures import ThreadPoolExecutor
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
import pickle


class BASERetrieverBuilder:
    def __init__(self, vector_path, vector_json_path=None):

        self.vector_path = vector_path
        self.vector_json_path = vector_json_path or f"{vector_path}/vector_store.json"

        self.build_map = {
            "pdf": self.build_pdf,
            "web": self.build_web,
        }
        self.chunk_map = {}

        if not os.path.exists(self.vector_json_path):
            os.makedirs(self.vector_path, exist_ok=True)
            self.index_info = {}
        else:
            try:
                with open(self.vector_json_path, "r", encoding="utf-8") as f:
                    self.index_info = json.load(f)
                
            except json.JSONDecodeError:
                print(f"[RetrieverBuilder][init] Wrong JSON format: {self.vector_json_path}, `index_info` will be empty.")
                self.index_info = {}

    def build(self, config: RetrieverConfig):

        if config.source_type in self.build_map:
            embeddings_model = self._embeddings_model(config.embeddings_model_id)

            faiss_path_list = self._check_local_faiss(config.source_type, config.name)
            if faiss_path_list and not config.update:
                faiss_path = faiss_path_list[str(config.max_depth)]
                return self._load_local_faiss(faiss_path, embeddings_model)
            else:
                return self.build_map[config.source_type](config, embeddings_model)
        else:
            raise ValueError(f"[RetrieverBuilder][build] Unknown source type: {config.source_type}")

    def build_pdf(self, config: RetrieverConfig, embeddings_model):
        loader = PyPDFDirectoryLoader(path=config.data_path)
        docs = loader.load() # 返回 Document 对象列表

        return self._chunking_build_retriever(docs, embeddings_model, config)

    def build_web(self, config: RetrieverConfig, embeddings_model):
        loader = WebBaseLoader(config.data_path, requests_per_second=config.requests_per_second, continue_on_failure=config.continue_on_failure)
        docs = loader.load() # 返回 Document 对象列表

        sub_links = self._extract_sublinks(config.data_path, config.max_depth)
        if sub_links:
            sub_docs = WebBaseLoader(list(sub_links), requests_per_second=config.requests_per_second, continue_on_failure=config.continue_on_failure).load()
            docs += sub_docs

        return self._chunking_build_retriever(docs, embeddings_model, config)

    def _embeddings_model(self, embeddings_model_id):
        from langchain_huggingface import HuggingFaceEmbeddings

        try:
            return HuggingFaceEmbeddings(model_name=embeddings_model_id)
        except Exception as e:
            raise ValueError(f"[RetrieverBuilder][_embeddings_model] Embeddings model not found: {embeddings_model_id}: {e}") from e
        
    def _check_local_faiss(self, source_type, name):
        if source_type in self.index_info and name in self.index_info[source_type]: # 已存储的向量库
            faiss_path_list = self.index_info[source_type][name]["faiss_path"]
            return faiss_path_list
        return False
    
    def _load_local_faiss(self, faiss_path, embeddings_model):
        vector = FAISS.load_local(faiss_path, embeddings_model, allow_dangerous_deserialization=True)
        retriever = vector.as_retriever()
        print(f"[RetrieverBuilder] Load FAISS index from: {faiss_path}")
        return retriever

    def _chunking_build_retriever(self, docs, embeddings_model, config: RetrieverConfig):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # TODO: 优化分块策略
        # 定长
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # 句尾

        documents = text_splitter.split_documents(docs)

        vector = FAISS.from_documents(documents, embeddings_model)
        self._save_local_faiss(vector, config)

        retriever = vector.as_retriever()
        return retriever

    def _extract_sublinks(self, data_path, max_depth):
        from urllib.parse import urljoin
        from bs4 import BeautifulSoup
        from collections import deque
        import requests
        

        queue = deque([(data_path, 1)])
        visited = set()
        sub_links = set()
        while queue:
            url, depth = queue.popleft()
            if depth > max_depth or url in visited:
                continue
            visited.add(url)
            try:
                # 获取 BeautifulSoup 对象 —— 可以操作 HTML 的结构化对象
                soup = BeautifulSoup(requests.get(url, timeout=5).content, 'html.parser')
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip() # 对 soup 对象的结构化操作
                    if not href or href.startswith(("#", "javascript:")): # 过滤空链接和 javascript 链接
                        continue
                    full_url = urljoin(url, href)
                    if (full_url not in visited
                        and full_url.startswith(data_path)
                        and not full_url.endswith((
                            ".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".pdf"
                        ))
                    ):
                        sub_links.add(full_url)
                        queue.append((full_url, depth + 1))
            except Exception as e:
                print(f"[RetrieverBuilder][_extract_sublinks] Failed to load URL: {url}, error: {e}")
                continue

            print(f"...Extracted {len(sub_links)} sub-links from {data_path}")
        return sub_links
    
    def _save_local_faiss(self, vector: VectorStore, config: RetrieverConfig):
        name = config.name
        source_type = config.source_type

        faiss_path = f"{self.vector_path}/{name}/"
        if config.max_depth: faiss_path += f"depth{config.max_depth}" # 如设置了 max_depth 则需分开保存

        vector.save_local(faiss_path)
        print(f"[RetrieverBuilder][_save_local_faiss] FAISS Index saved: {faiss_path}")

        if source_type not in self.index_info:
            self.index_info[source_type] = {}
        
        if name not in self.index_info[source_type]:
            self.index_info[source_type][name] = config.index_saving_dict()

        if "faiss_path" not in self.index_info[source_type][name]:
            self.index_info[source_type][name]["faiss_path"] = {}

        if faiss_path not in self.index_info[source_type][name]["faiss_path"]:
            self.index_info[source_type][name]["faiss_path"][str(config.max_depth)] = faiss_path

        with open(self.vector_json_path, "w", encoding="utf-8") as f:
            json.dump(self.index_info, f, indent=2, ensure_ascii=False)
        print(f"[RetrieverBuilder][_save_local_faiss] Vector JSON updated: {self.vector_json_path}")

class RetrieverBuilder:
    """
    Args:
        vector_path: 向量库目录地址, 默认为空
        vector_json_path: 文档目录地址, 默认为空
    """
    def __init__(self, vector_path="./vector_store", vector_json_path=None):
        self.vector_path = vector_path
        self.vector_json_path = vector_json_path or f"{vector_path}/vector_store.json"

        if not os.path.exists(self.vector_json_path):
            os.makedirs(self.vector_path, exist_ok=True)
            print(f"[RetrieverBuilder][init] No Vector JSON: {self.vector_json_path}, `index_info` will be empty.")
            self.index_info = {}
        else:
            try:
                with open(self.vector_json_path, "r", encoding="utf-8") as f:
                    self.index_info = json.load(f)
            except json.JSONDecodeError:
                print(f"[RetrieverBuilder][init] Cannot open Vector JSON: {self.vector_json_path}, `index_info` will be empty.")
                self.index_info = {}

        self.build_map = {
            "pdf": self.build_pdf,
            "web": self.build_web,
        }

    def build(self, config: RetrieverConfig):
        """创建检索器"""

        if config.source_type in self.build_map:
            embeddings_model = self._embeddings_model(config.embeddings_model_id)

            # 若不要求重载，且存在本地索引，则直接加载
            index_path = self._index_dir(config)
            if os.path.exists(index_path) and self._check_index(index_path):
                if not config.update:
                    return self._load_local_hybrid(index_path, embeddings_model)

            return self.build_map[config.source_type](config, embeddings_model)
        else:
            raise ValueError(f"[RetrieverBuilder][build] Unknown source type: {config.source_type}")

    def build_pdf(self, config: RetrieverConfig, embeddings_model):
        """创建PDF检索器"""

        loader = PyPDFDirectoryLoader(path=config.data_path)
        docs = loader.load()
        return self._hybrid_build_retriever(docs, embeddings_model, config)

    def build_web(self, config: RetrieverConfig, embeddings_model):
        """创建网页检索器"""

        loader = WebBaseLoader(config.data_path, requests_per_second=config.requests_per_second, continue_on_failure=config.continue_on_failure)
        docs = loader.load()

        sub_links = self._extract_sublinks(config.data_path, config.max_depth)
        if sub_links:
            with ThreadPoolExecutor(max_workers=8) as executor:
                sub_docs_list = list(executor.map(lambda url: WebBaseLoader(url).load(), sub_links))
            for sub_doc in sub_docs_list:
                docs.extend(sub_doc)

        return self._hybrid_build_retriever(docs, embeddings_model, config)
    
    def _embeddings_model(self, embeddings_model_id):
        """加载嵌入模型"""

        try:
            embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_id)
            print(f"[RetrieverBuilder][_embeddings_model] Load embeddings model: {embeddings_model_id}")
            return embeddings_model
        except Exception as e:
            raise ValueError(f"[RetrieverBuilder][_embeddings_model] Embeddings model not found: {embeddings_model_id}: {e}") from e
    
    def _check_index(slef, index_path):
        """检查向量索引是否存在"""

        return os.path.exists(f"{index_path}/index.faiss") and os.path.exists(f"{index_path}/index.pkl") and os.path.exists(f"{index_path}/bm25.pkl")

    def _load_local_hybrid(self, index_path, embeddings_model):
        """加载本地混合检索器"""

        vector = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
        print(f"[RetrieverBuilder][_load_local_hybrid] Load faiss vector from: {index_path}")
            
        with open(f"{index_path}/bm25.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        print(f"[RetrieverBuilder][_load_local_hybrid] Load bm25 from: {index_path}")

        hybrid_retriever = EnsembleRetriever(retrievers=[vector.as_retriever(), bm25_retriever], weights=[0.75, 0.25])
        print(f"[RetrieverBuilder][_load_local_hybrid] Load hybrid retriever from: {index_path}")
        return hybrid_retriever
    
    def _extract_sublinks(self, data_path, max_depth):
        """提取子链接"""

        from urllib.parse import urljoin
        from bs4 import BeautifulSoup
        from collections import deque
        import requests

        queue = deque([(data_path, 1)])
        visited = set()
        sub_links = set()
        while queue:
            url, depth = queue.popleft()
            if depth > max_depth or url in visited:
                continue
            visited.add(url)
            try:
                soup = BeautifulSoup(requests.get(url, timeout=5).content, 'html.parser')
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if not href or href.startswith(("#", "javascript:")):
                        continue
                    full_url = urljoin(url, href)
                    if (full_url not in visited and full_url.startswith(data_path) and not full_url.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".pdf"))):
                        sub_links.add(full_url)
                        queue.append((full_url, depth + 1))
            except Exception as e:
                print(f"[RetrieverBuilder][_extract_sublinks] Failed to load URL: {url}, error: {e}")
                continue
        print(f"...Extracted {len(sub_links)} sub-links from {data_path}")
        return sub_links

    def _hybrid_build_retriever(self, docs, embeddings_model, config: RetrieverConfig):
        """构建混合检索器"""

        doc_chunks = self._chunk_document(docs, config.source_type)

        if config.update and os.path.exists(self._doc_path(config)):
            print("[RetrieverBuilder][_hybrid_build_retriever] Incremental update docstore...")

            # 添加新 doc (使用 add_documents 方法，也可以构造新向量库再用 merge_from 方法)
            new_docs = self._get_new_documents(config, doc_chunks)
            vector = FAISS.load_local(self._index_dir(config), embeddings_model, allow_dangerous_deserialization=True)
            vector.add_documents(new_docs)
        else:
            print("[RetrieverBuilder][_hybrid_build_retriever] Covering docstore...")
            vector = FAISS.from_documents(doc_chunks, embeddings_model)

        bm25_retriever = BM25Retriever.from_documents(doc_chunks)
        hybrid_retriever = EnsembleRetriever(retrievers=[vector.as_retriever(), bm25_retriever], weights=[0.75, 0.25])

        print("[RetrieverBuilder][_hybrid_build_retriever] Hybrid Retriever Built...")
        self._save_local_hybrid(vector, bm25_retriever, config)
        return hybrid_retriever

    def _chunk_document(self, docs, source_type):
        """切分文档"""

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", " "]
        )

        if source_type == "web":
            from langchain_text_splitters import HTMLHeaderTextSplitter

            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header1"), ("h2", "Header2")])
            html_chunks = []
            for doc in docs:
                chunks = html_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    chunk.metadata = doc.metadata  # 保留元数据 ['source', 'title', 'description', 'language']
                html_chunks.extend(chunks)

            final_docs = text_splitter.split_documents(html_chunks)

        elif source_type == "pdf":
            final_docs = text_splitter.split_documents(docs)

        print(f"[RetrieverBuilder][_chunk_document] {len(final_docs)} chunks created")

        return final_docs

    def _get_new_documents(self, config, docs):
        """获取新文档"""

        doc_path = self._doc_path(config)
        with open(doc_path, "rb") as f:
            old_docs = pickle.load(f)
        old_set = set([d.page_content for d in old_docs])
        new_docs = [d for d in docs if d.page_content not in old_set]
        # with open(doc_path, "wb") as f:
        #     pickle.dump(docs, f)
        return new_docs

    def _save_local_hybrid(self, vector, bm25_retriever, config):
        """保存混合检索器到本地"""

        index_path = self._index_dir(config)
        vector.save_local(index_path)
        with open(f"{index_path}/bm25.pkl", "wb") as f:
            pickle.dump(bm25_retriever, f)
        if config.save_data:
            with open(self._doc_path(config), "wb") as f:
                pickle.dump(vector.docstore, f)
        print(f"[RetrieverBuilder][_save_local_hybrid] Index saved to {index_path}...")
        self._update_index_info(config, index_path)

    def _index_dir(self, config):
        """获取/指定本地向量索引目录"""

        source_type, name, depth = config.source_type, config.name, config.max_depth

        if self.index_info.get(source_type, {}).get(name, {}).get("index_path", {}): # base/source/name
            # print(self.index_info[source_type][name]["index_path"], depth)
            path = self.index_info[source_type][name]["index_path"][str(depth)]
        else:
            path = f"{self.vector_path}/{source_type}/{name}"

        return path


    def _doc_path(self, config):
        """获取原文档 pkl 路径"""

        return f"{self._index_dir(config)}/docs.pkl"

    def _update_index_info(self, config, index_path):
        """更新索引信息
        **
            source1: 
                name1:
                    data_path: ...,
                    index_path: 
                        0: ...,
                        1: ...,
                ,
                ...
        """

        source_type, name, depth = config.source_type, config.name, config.max_depth

        if source_type not in self.index_info:
            self.index_info[source_type] = {}
        if name not in self.index_info[source_type]:
            self.index_info[source_type][name] = config.index_saving_dict()

        if not self.index_info[source_type][name].get("index_path"):
            self.index_info[source_type][name]["index_path"] = {depth: index_path}
        else:
            self.index_info[source_type][name]["index_path"][depth] = index_path

        with open(self.vector_json_path, "w", encoding="utf-8") as f:
            json.dump(self.index_info, f, indent=2, ensure_ascii=False)

        print(f"[RetrieverBuilder][_update_index_info] Index info update: {self.vector_json_path}")



if __name__ == "__main__":
    # 切换运行路径
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from configs import VECTOR_STORE_PATH
    builder = RetrieverBuilder(VECTOR_STORE_PATH)
    
    retriever_config = RetrieverConfig(
        source_type="web",
        name="mcp",
        data_path="https://modelcontextprotocol.io/introduction",
        max_depth=1,
        save_data=False,
        update=False
    )
    # print(retriever_config.__dict__)

    retriever = builder.build(retriever_config)
    results = retriever.invoke("What is Model Content Protocal?")
    print(f"已找到结果：{results[0].page_content[0:10]} ..." if results else "未找到结果")

    

    retriever_config = RetrieverConfig(
        source_type="pdf",
        name="6402",
        save_data=True,
        data_path=r"E:/NTU/term 2/6402",
    )
    # print(retriever_config.__dict__)

    retriever = builder.build(retriever_config)
    results = retriever.invoke("What is Mid-tread Quantizer?")
    print(f"已找到结果：{results[0].page_content[0:10]} ..." if results else "未找到结果")
