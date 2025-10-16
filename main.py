import os
from langsmith import traceable
from dotenv import load_dotenv
from rich.prompt import Prompt
from rich.style import Style
from rich.rule import Rule
import time
from configs import AgentConfig, RetrieverConfig, VECTOR_STORE_PATH, ENABLE_TRACE
from agent import RichStreamingCallback, TAgent

os.environ["HF_ENDPOINT"] = "www.hf-mirror.com"
os.environ["HF_HOME"] = "E:\Production\models\hgf"


load_dotenv()

@traceable(enable=ENABLE_TRACE)
def main():
    config = AgentConfig(llm_id="deepseek-chat", verbose=False, streaming=True, memory_mgmt="window", use_search=False, use_retriever=True,
        retriever_config=[
            RetrieverConfig(
                source_type="pdf",
                name="6402",
                save_data=True,
                data_path=r"E:/NTU/term 2/6402",
            ),
            RetrieverConfig(
                source_type="pdf",
                name="6403",
                save_data=True,
                data_path=r"E:/NTU/term 2/6403",
            ),
            RetrieverConfig(
                source_type="pdf",
                name="7207",
                save_data=True,
                data_path=r"E:/NTU/term 2/7207",
            )
        ],
    )
    callback = RichStreamingCallback()

    start = time.time()
    tagent = TAgent(config, callbacks=[callback], vector_path=VECTOR_STORE_PATH)
    elapsed  = time.time() - start
    
    console = callback.console

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
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
