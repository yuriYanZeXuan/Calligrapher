import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

client = OpenAI(
    api_key="MAASace45968cdbf4afeb71d07ecef846c94",
    base_url=os.getenv("QST_BASE_URL")
)

# Example 2: 非流式调用 LLM 服务
completion = client.chat.completions.create(
    model="qwen3-vl-235b-a22b-instruct",  # 在 Body 中指明要访问的模型名
    messages=[
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "你好"}
    ],
    stream=False,
    max_tokens=4096,
    temperature=0.9
)

print(completion.model_dump_json())
# {"id":"chatcmpl-98643f5ab6f727b05074a819eec770b3","choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"你好！有什么我可以帮你的吗？😊","refusal":null,"role":"assistant","annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"stop_reason":null,"token_ids":null}],"created":1769939382,"model":"qwen3-vl-235b-a22b-instruct","object":"chat.completion","service_tier":null,"system_fingerprint":null,"usage":{"completion_tokens":10,"prompt_tokens":18,"total_tokens":28,"completion_tokens_details":null,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
