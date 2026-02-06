import os
import json
import base64
import requests
import argparse
from pathlib import Path

def get_api_key():
    """获取 API Key，优先级: GEMINI_IMAGE_API_KEY > RUNWAY_API_KEY"""
    return "a938bc0e9989472aaea5f136153dee35"

def encode_image(image_path):
    """读取图片并转换为 Base64 编码"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    mime_type = "image/png"
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
        
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(path.read_bytes()).decode('utf-8')
        }
    }

def generate_content(prompt, image_path, output_prefix="output"):
    """调用 Gemini API 生成图片和思考过程"""
    api_key = get_api_key()
    if not api_key:
        print("错误: 未找到 API Key (GEMINI_IMAGE_API_KEY 或 RUNWAY_API_KEY)")
        return

    endpoint = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
    
    # 构造请求体
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                encode_image(image_path)
            ]
        }],
        "generationConfig": {
            "temperature": 1,
            "maxOutputTokens": 32768,
            "responseModalities": ["TEXT", "IMAGE"],
            "topP": 0.95,
            "imageConfig": {
                "aspectRatio": "1:1",
                "imageSize": "1K",
                "imageOutputOptions": {
                    "mimeType": "image/png"
                },
                "personGeneration": "ALLOW_ALL"
            }
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "OFF"
            }
        ]
    }
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    print("正在请求 Gemini API...")
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=1200)
        response.raise_for_status()
        result = response.json()
        
        # 处理响应
        candidates = result.get("candidates", [])
        if not candidates:
            print("未收到有效响应候选")
            return

        parts = candidates[0].get("content", {}).get("parts", [])
        
        # 保存思考过程和图片
        for i, part in enumerate(parts):
            # 处理文本 (思考过程)
            if "text" in part:
                text_content = part["text"]
                text_file = f"{output_prefix}_thought.txt"
                with open(text_file, "a+", encoding="utf-8") as f:
                    f.write(text_content)
                print(f"思考过程已保存至: {text_file}")
                
            # 处理图片
            if "inlineData" in part:
                img_data = part["inlineData"]["data"]
                img_bytes = base64.b64decode(img_data)
                img_file = f"{output_prefix}_image.png"
                with open(img_file, "wb") as f:
                    f.write(img_bytes)
                print(f"生成的图片已保存至: {img_file}")

    except Exception as e:
        print(f"发生错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"API 响应: {e.response.text}")

if __name__ == "__main__":

    a="理解图片中HiCache的实现原理，然后生成一个示意原理的学术论文配图"

    parser = argparse.ArgumentParser(description="Gemini Image Generation Demo")
    parser.add_argument("--image_path",default="/Users/yanzexuan/code/CTF_overleaf/hicache.png", help="参考图片的路径")
    parser.add_argument("--prompt", default=a, help="提示词")
    parser.add_argument("--output", default="./nanobanana_test", help="输出文件前缀")
    
    args = parser.parse_args()
    
    generate_content(args.prompt, args.image_path, args.output)
