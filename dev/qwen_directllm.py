import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

client = OpenAI(
    api_key=os.getenv("QST_API_KEY"),
    base_url=os.getenv("QST_BASE_URL")
)

# Example 2: 非流式调用 LLM 服务
completion = client.chat.completions.create(
    model="qwen2.5-vl-32b-instruct",  # 在 Body 中指明要访问的模型名
    messages=[
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "帮我制定一份日本的五天四夜的旅游攻略，小红书风格"}
    ],
    stream=False,
    max_tokens=4096,
    temperature=0.9
)

print(completion.model_dump_json())
