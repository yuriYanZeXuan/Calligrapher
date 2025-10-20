import os
import config
from openai import OpenAI
import tqdm

def generate_instructions(output_path: str, total_instructions: int = 20000, batch_size: int = 100, service: str = "remote"):
    """
    Generates instructions using an LLM and saves them to a file.

    Args:
        output_path (str): The path to save the generated instructions.
        total_instructions (int): The total number of instructions to generate.
        batch_size (int): The number of instructions to generate per API call.
        service (str): The service to use ('local' or 'remote').
    """
    try:
        service_config = config.SERVICES[service]
        
        base_url = service_config["api_base_url"]

        client = OpenAI(
            base_url=base_url,
            api_key=service_config["api_key"]
        )

        system_prompt = "You are a helpful assistant specialized in generating creative and diverse instructions for an image generation model. The model is designed to create images containing text. Your task is to generate a list of instructions. Each instruction should be a short, single-line description of an image that includes some text. The text to be included in the image should be enclosed in single quotes."
        user_prompt_template = f"Please generate {batch_size} diverse instructions for a text-to-image model. The instructions should be in a mix of Chinese, English, Japanese, and Korean. The text content within the quotes should be simple and common words or short phrases. The overall instruction should be concise.\n\nHere are some examples of the desired format and style:\n- A sign that says 'Hello World' in a futuristic font.\n- Graffiti of the word 'ART' on a brick wall, vibrant colors.\n- 一个用未来感字体写的“你好，世界”的标志。\n- '어서오세요' written on a welcome mat in a friendly Korean font.\n- 「こんにちは」と書かれた木の看板。\n\nPlease provide each instruction on a new line, without any numbering or bullet points."

        num_iterations = total_instructions // batch_size
        print(f"Generating {total_instructions} instructions in {num_iterations} batches of {batch_size}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            for _ in tqdm.tqdm(range(num_iterations)):
                response = client.chat.completions.create(
                    model=service_config["llm_model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_template},
                    ]
                )
                
                generated_text = response.choices[0].message.content.strip()
                instructions = [line.strip() for line in generated_text.split('\n') if line.strip()]
                
                for instruction in instructions:
                    f.write(instruction + '\n')
        
        print(f"Successfully generated and saved {total_instructions} instructions to {output_path}")

    except Exception as e:
        print(f"An error occurred during instruction generation: {e}")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate instructions with specified quantity.")
    parser.add_argument("--total", type=int, default=2000, help="Total number of instructions to generate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size per API call")
    args = parser.parse_args()

    output_filename = f"instructions_{args.total}_generated.txt"
    output_file_path = os.path.join(os.path.dirname(__file__), output_filename)

    generate_instructions(output_file_path, total_instructions=args.total, batch_size=args.batch_size)
