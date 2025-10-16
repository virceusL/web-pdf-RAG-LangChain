
import os
import time
from typing import Any
from rich.style import Style
from dotenv import load_dotenv
from configs import AgentConfig, RetrieverConfig
from langchain_core.messages import BaseMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()


class RichStreamingCallback(StreamingStdOutCallbackHandler):
    def __init__(self):
        from rich.console import Console

        self.console = Console(width=120)
        self._start_time = None
        self.first_token_received = False
        self.first_token_latency = None  # 对外

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        self._start_time = time.time()
        self.first_token_received = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:

        if not self.first_token_received and self._start_time and token:
            self.first_token_latency = time.time() - self._start_time
            self.console.print(f"[bold green]Agent>[/] ", style=Style(color="bright_cyan"), end="")
            self.first_token_received = True
            self._start_inference = True

        self.console.print(token, style="bright_cyan", end="")

    def on_llm_end(self, response, **kwargs) -> None:
        if not self.first_token_received: return
        self.console.print()  # inference中, 打印换行, 结束后换一行


class TAgent:
    def __init__(self, config: AgentConfig, callbacks: list, vector_path: str):
        from retriever_builder import RetrieverBuilder
        
        self.config = config
        self.callbacks = callbacks
        self.retriever_builder = RetrieverBuilder(vector_path)

        self.llm = self._init_llm()
        self.tools = self._init_tools()
        self.prompt = self._init_prompt()
        self.agent = self._init_agent()
        self.memory = self._init_memory()

        self.executor = self._init_executor()
    
    def _init_llm(self):
        """初始化LLM"""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.config.llm_id,
            temperature=self.config.temp,
            verbose=self.config.verbose,
            streaming=self.config.streaming,
            callbacks=self.callbacks
        )

    def _init_tools(self):
        """初始化工具"""
        from langchain.tools.retriever import create_retriever_tool

        tools = []
        if self.config.use_retriever and self.config.retriever_config:
            for rtv_cfg in self.config.retriever_config:
                retriever = self.retriever_builder.build(rtv_cfg)
                cur_tool = create_retriever_tool(
                    retriever=retriever,
                    name=rtv_cfg.name,
                    description=rtv_cfg.description
                )
                tools.append(cur_tool)

        # 其他工具（示例）
        if self.config.use_search:
            from langchain_community.tools import TavilySearchResults
            search_tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=True,
                # include_domains=[...],
                # exclude_domains=[...],
                # name="...",            # overwrite default tool name
                # description="...",     # overwrite default tool description
                # args_schema=...,       # overwrite default args_schema: BaseModel
            )
            tools.append(search_tool)

        return tools
    
    def _init_prompt(self):
        """初始化Prompt"""
        from langchain import hub
        from langchain_core.prompts.chat import MessagesPlaceholder

        self.enable_chat_history = False
        prompt = hub.pull(self.config.prompt_id)
        for message in prompt.messages:
            if isinstance(message, MessagesPlaceholder) and message.variable_name == 'chat_history':
                self.enable_chat_history = True
                break
        return prompt

    def _init_agent(self):
        """初始化Agent"""
        from langchain.agents import create_tool_calling_agent

        return create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )
    
    def _init_memory(self):
        """初始化Memory管理组件"""
        from langchain_openai import ChatOpenAI
        
        if self.config.memory_mgmt and self.enable_chat_history:
            if self.config.memory_mgmt == "window":
                from langchain.memory import ConversationBufferWindowMemory
                return ConversationBufferWindowMemory(
                    memory_key="chat_history",  # 在prompt中对应的占位符变量名
                    return_messages=True,       # 以Message对象格式返回历史（而非字符串）
                    k=5,                        # 每次返回的历史记录条数, 默认值为5
                )
            elif self.config.memory_mgmt == "summary":
                from langchain.memory import ConversationSummaryMemory
                sum_llm = ChatOpenAI(model=self.config.llm_id, temperature=0)
                return ConversationSummaryMemory(
                    llm=sum_llm,
                    memory_key="chat_history",
                    max_token_limit=256,        # 每次返回的摘要最大token数
                    return_messages=True, 
                )
            elif self.config.memory_mgmt == "full":
                from langchain.memory import ConversationBufferMemory
                return ConversationBufferMemory(
                    memory_key="chat_history",  # 在prompt中对应的占位符变量名
                    max_token_limit=1024,       # 历史记录最大token数  
                    return_messages=True,
                )
        return None

    def _init_executor(self):
        """初始化Executor"""
        from langchain.agents import AgentExecutor
        
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.config.verbose,
            memory=self.memory,
            max_iterations=5,
            timeout=30,
        )



from langsmith import traceable
# @traceable(enable=ENABLE_TRACE)
def main():
    
    from rich.rule import Rule
    from rich.prompt import Prompt


    
    config = AgentConfig(
        llm_id="deepseek-chat",
        verbose=False,
        streaming=True,
        memory_mgmt="window",
        use_search=False,
        use_retriever=True,
        retriever_config=[
            # RetrieverConfig(
            #     name="langchain",
            #     data_path="https://docs.smith.langchain.com/",
            #     max_depth=3
            # ),
            RetrieverConfig(
                source_type="pdf",
                name="6402",
                data_path=r"E:/NTU/term 2/6402",
            ),
            RetrieverConfig(
                source_type="pdf",
                name="6403",
                data_path=r"E:/NTU/term 2/6403",
            ),
            RetrieverConfig(
                source_type="pdf",
                name="7207",
                data_path=r"E:/NTU/term 2/7207",
            )
        ],
    )
    from configs import VECTOR_STORE_PATH
    
    callback = RichStreamingCallback()

    start = time.time()
    tagent = TAgent(config, callbacks=[callback], vector_path=VECTOR_STORE_PATH)
    elapsed  = time.time() - start
    
    console = callback.console
    console.print(f"TOOLS: {tagent.tools}", style=Style(color="grey74", italic=True))
    console.print(f"[INIT TIME] {elapsed:.2f} seconds", style=Style(color="grey74", italic=True))
    console.print(Rule(style="grey50",align="center"))

    # # warmup
    # tagent.executor.invoke({"input": "Hi. ##THIS IS A WARMUP MESSAGE. DO NOT OUTPUT A SINGLE WORD. DO NOT RESPONSE. DO NOT CONFIRM."})

    while True:

        query = Prompt.ask(prompt="[bold green]User>[/]", console=console, default="exit/quit", show_default=True)
        if query.lower() in ["exit", "quit"]: break

        # with Status("[cyan]...[/]", spinner="dots", console=console) as status:
        start = time.time()
        response = tagent.executor.invoke({"input": query})
        elapsed  = time.time() - start

        # console.print(f"[bold green]Agent>[/] {response['output']}", style=Style(color="bright_cyan"))

        ttft = f"{callback.first_token_latency:.2f} seconds" if callback.first_token_received else "N/A"
        tte = f"{elapsed:.2f} seconds"

        console.print(f"[LATENCY] TTFT: {ttft}, TTE: {tte}, [RUN TRACE]: https://smith.langchain.com/", style=Style(color="grey74", italic=True))
        console.print(Rule(style="grey50",align="center"))

if __name__ == "__main__":
    # 切换运行路径
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()