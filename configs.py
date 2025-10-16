from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

ENABLE_TRACE=True # Enable LangSmith Tracing
VECTOR_STORE_PATH = "./test_save" # Path to save FAISS index
# ACCEPT_SOURCE_TYPES = ["pdf", "web"]


@dataclass
class RetrieverConfig:
    """
    Args:
        name: 文档名, 默认为空
        data_path: 文档目录或网页地址, 默认为空
        save_data: 是否保存读取到的文档, 默认为 False
        source_type: 文档类型, 默认为空
        description: 当前描述, 默认为空
        embeddings_model_id: 嵌入模型 ID, 可替换其他 hgf 嵌入模型
        max_depth: 创建 WebRetriver 时提取子链接的最大深度, 默认为 0, 仅 `source_type` 为 `web` 时设置为1以上
        update: 是否更新现存 retriever, 默认为 False, 即从本地 pickle 文件导入并更新（若存在）
        requests_per_second: 每秒请求数, 默认为 2, 仅 `source_type` 为 `web` 时有效
        continue_on_failure: 是否在失败时继续, 默认为 True, 仅 `source_type` 为 `web` 时有效
    """
    # 调用层次：RetrieverBuilder <- Cfg.name, path, 
    def __init__(self,
                name: str="",
                data_path: str="",
                save_data: bool=False,
                source_type: Literal["pdf", "web"] = "",
                description: str="",
                embeddings_model_id: str="sentence-transformers/distiluse-base-multilingual-cased-v2",
                max_depth: int=0,
                update=False,
                requests_per_second=2,
                continue_on_failure=True
    ):
        if source_type not in ["pdf", "web"]:
            raise ValueError("source_type must be 'pdf' or 'web'")
        if max_depth < 0 or (source_type == "web" and max_depth == 0):
            raise ValueError("max_depth must be greater than ('web') or equal to 0 ('pdf')")

        self.name = name
        self.data_path = data_path
        self.save_data = save_data
        self.source_type = source_type
        if description:
            self.description = description
        else:
            self.description = f"Use this tool to retrieve any content of {self.name}. Input should be query."
        self.embeddings_model_id = embeddings_model_id
        self.max_depth = max_depth
        self.update = update
        self.requests_per_second = requests_per_second
        self.continue_on_failure = continue_on_failure

    def index_saving_dict(self):
        return {
            "data_path": self.data_path,
        }

    def to_dict(self):
        return self.__dict__

@dataclass
class AgentConfig:
    """
    Args:
        llm_id: LLM 模型ID, 默认 deepseek-chat
        temp: LLM 温度, 默认 0.7
        verbose: LLM Calls 是否打印详细信息, 默认 False
        streaming: 是否流式返回, 默认 False
        memory_mgmt: 内存管理策略, 可选 {"window", "summary", "full"}
        prompt_id: 提示模板ID, 默认 openai-functions-agent 格式
        use_search: 是否使用 Tavily 搜索器, 默认 False
        use_retriever: 是否使用自定义检索器, 默认 False
        retriever_config: 包含检索器配置的列表 [RetrieverConfig1, RetrieverConfig2, ...]
    """
    def __init__(self,
                llm_id: str="deepseek-chat",
                temp: float=0.7,
                verbose=False,
                streaming: bool=False,
                memory_mgmt: Literal["window", "summary", "full"] = "full",
                prompt_id: str="hwchase17/openai-functions-agent",
                use_search: bool=True,
                use_retriever: bool=False,
                retriever_config: list[RetrieverConfig] = [],
    ):
        self.llm_id = llm_id
        self.temp = temp
        self.verbose = verbose
        self.streaming = streaming
        self.memory_mgmt = memory_mgmt
        self.prompt_id=prompt_id
        self.use_search = use_search
        self.use_retriever = use_retriever
        self.retriever_config = retriever_config
        
    def to_dict(self):
        return self.__dict__