
# gpt系列api使用方法：

## gpt 5.4
curl --location --output output.json.gz \
'https://maas.devops.xiaohongshu.com/runway/global/openai/chat/completions?api-version=2024-12-01-preview' \
--header 'api-key: QSTfba6e960fead06bf5d253e7c72db4581' \
--header 'Content-Type: application/json' \
--data '{
     "model": "gpt-5.4",
    "messages":[
        {
            "role":"system",
            "content":"You are a helpful assistant."
        },
        {
            "role":"user",
            "content":"你好"
        }
    ]
}' && gzip -dc output.json.gz

> {"choices":[{"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"protected_material_code":{"detected":false,"filtered":false},"protected_material_text":{"detected":false,"filtered":false},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"finish_reason":"stop","index":0,"logprobs":null,"message":{"annotations":[],"content":"你好！有什么我可以帮你的吗？","refusal":null,"role":"assistant"}}],"created":1777977845,"id":"chatcmpl-Dc7XxhOQzus4wK9iiRv4IHVdTDMRf","model":"gpt-5.4-2026-03-05","object":"chat.completion","prompt_filter_results":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"jailbreak":{"detected":false,"filtered":false},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}],"service_tier":"default","system_fingerprint":null,"usage":{"completion_tokens":13,"completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":0,"rejected_prediction_tokens":0},"latency_checkpoint":{"engine_tbt_ms":12,"engine_ttft_ms":113,"engine_ttlt_ms":268,"pre_inference_ms":43,"service_tbt_ms":12,"service_ttft_ms":235,"service_ttlt_ms":380,"total_duration_ms":348,"user_visible_ttft_ms":191},"prompt_tokens":17,"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0},"total_tokens":30}}

## gpt 4o
### 多图像输入
curl --location 'https://runway.devops.rednote.life/openai/chat/completions?api-version=2024-12-01-preview' \
--header 'api-key: {$your_key}' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ]
        }
    ]
}'

### base64编码图像
curl --location 'https://runway.devops.rednote.life/openai/chat/completions?api-version=2024-12-01-preview' \
--header 'api-key: {$your_key}' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]
}'

# gemini系列api使用方法

curl --location 'https://runway.devops.rednote.life/openai/google/v1:generateContent' \
--header 'api-key: {your_key}' \
--header 'Content-Type: application/json' \
--data '{
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": "组织一场游学"
                }
            ]
        }
    ],
    "systemInstruction": {
        "parts": [
            {
                "text": "你是一名老师"
            }
        ]
    },
    "generationConfig": {
        "temperature": 1,
        "maxOutputTokens": 65535,
        "topP": 0.95,
        "seed": 0,
        "thinkingConfig": {
            "thinkingLevel": "HIGH",
            "includeThoughts":true
        }
    }
}'

## base64编码图像

mport requests
import json
import base64

url = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
image_path = "/Users/yangxudong1/PycharmProjects/worktest/7a69d33689e5be0afef9326be55d80a2-removebg-preview.png"
with open(image_path, 'rb') as f:
    image_data = f.read()
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
payload = json.dumps({
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "inlineData": {
            "mimeType": "image/png",
            "data": base64_encoded
          }
        },
        {
          "text": "图片里有什么"
        }
      ]
    }
  ],
  "generationConfig": {
    "thinkingConfig": {
      "includeThoughts": True
    }
  }
})
headers = {
  'api-key': '{your_key}',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

# kimi2.5系列api使用方法
from openai import OpenAI

client = OpenAI(
    api_key="QSTfba6e960fead06bf5d253e7c72db4581",  # 你在 QS 平台生成的 token
    base_url="https://maas.devops.xiaohongshu.com/v1",  # DirectLLM 域名
    default_headers={
        "x-maas-user-email": "yanzexuan@xiaohongshu.com",
        "x-maas-app-id": "qs-api"
    },
)


## Example : 非流式调用 LLM 服务
completion = client.chat.completions.create(
    model="kimi-k2.5",  # 在 Body 中指明要访问的模型名
    messages=[
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "帮我制定一份日本的五天四夜的旅游攻略，小红书风格"}
    ],
    stream=False,
    max_tokens=4096,
    temperature=0.9,
      chat_template_kwargs={
          "thinking": True,
          "enable_thinking": True
      }
)

print(completion.model_dump_json())