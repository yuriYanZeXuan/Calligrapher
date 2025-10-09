import requests
import json
import os
import pdb
import base64
from openai import OpenAI



def call_doubao4(prompt):
    client = OpenAI(
        base_url="http://35.220.164.252:3888/v1",
        api_key=os.environ.get("API_KEY", "")
    )

    imagesResponse = client.images.generate( 
        model="doubao-seedream-4-0-250828", 
        prompt="星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车，抢视觉冲击力，电影大片，末日既视感，动感，对比色，oc渲染，光线追踪，动态模糊，景深，超现实主义，深蓝，画面通过细腻的丰富的色彩层次塑造主体与场景，质感真实，暗黑风背景的光影效果营造出氛围，整体兼具艺术幻想感，夸张的广角透视效果，耀光，反射，极致的光影，强引力，吞噬",
        size="1024x1024",
        response_format="url",
        extra_body={
            "watermark": False,
        },
    ) 
    
    import requests
    from pathlib import Path

    image_url = imagesResponse.data[0].url
    print(image_url)

    # 创建imgs目录（如果不存在）
    imgs_dir = Path("imgs")
    imgs_dir.mkdir(exist_ok=True)

    # 下载图片并保存
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        image_path = imgs_dir / "downloaded_image.png"
        with open(image_path, "wb") as f:
            f.write(image_response.content)
        print(f"图片已保存到: {image_path}")
    else:
        print(f"下载图片失败，状态码: {image_response.status_code}")

    return image_url


if __name__ == "__main__":
    prompt = "What is the capital of China?"

    # print(call_gpt(prompt))
    print(call_doubao4(prompt))