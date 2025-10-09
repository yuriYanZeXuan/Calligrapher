from openai import OpenAI
import config

client = OpenAI(
    base_url=config.API_BASE_URL,
    api_key=config.API_KEY
)

def simplify_prompt(augmented_prompt: str) -> str:
    """
    Uses an LLM to simplify an augmented prompt to extract the core text.
    For example, from "A colorful graffiti of the word 'NGANGUR' on a brick wall" -> "NGANGUR"

    Args:
        augmented_prompt (str): The complex prompt for the T2I model.

    Returns:
        str: The simplified text, or an empty string if an error occurs.
    """
    try:
        print(f"Simplifying prompt: '{augmented_prompt}'")
        
        system_prompt = "You are an expert in prompt engineering. Your task is to extract only the main text phrase or word that is intended to be written in the image, from the given prompt. Do not add any explanation or quotation marks."
        
        response = client.chat.completions.create(
            model=config.TEXT_GEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_prompt}
            ],
            temperature=0,
            max_tokens=50,
        )
        simplified_text = response.choices[0].message.content.strip()
        print(f"Simplified text: '{simplified_text}'")
        return simplified_text
    except Exception as e:
        print(f"An error occurred during prompt simplification: {e}")
        return ""

if __name__ == '__main__':
    # Example usage
    augmented_prompt = "A colorful graffiti of the word 'NGANGUR' on a brick wall"
    simplify_prompt(augmented_prompt)
    
    augmented_prompt_2 = "A neon sign that says 'Hello World'"
    simplify_prompt(augmented_prompt_2)
