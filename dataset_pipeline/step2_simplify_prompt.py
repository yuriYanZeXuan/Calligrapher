from openai import OpenAI
import config
import re

def simplify_prompt(prompt: str, service: str = "remote") -> str:
    """
    Uses an LLM to simplify a complex prompt into a concise text string,
    which is expected to be visually present in the generated image.
    
    Args:
        prompt (str): The complex input prompt.
        service (str): The service to use ('local' or 'remote').

    Returns:
        str: The simplified text, or None if an error occurs.
    """
    try:
        service_config = config.SERVICES[service]
        
        # Determine the correct base URL
        if service == "local":
            base_url = service_config["api_base_url_llm"]
        else: # remote
            base_url = service_config["api_base_url"]

        client = OpenAI(
            base_url=base_url,
            api_key=service_config["api_key"]
        )

        system_prompt = "You are a helpful assistant. Your task is to extract the exact text content enclosed in single quotes from the user's prompt. Respond with only the text content itself, without any additional explanation or formatting."
        user_prompt = f"Please extract the text from the following prompt: {prompt}"

        print(f"Simplifying prompt: '{prompt}' using '{service}' service at {base_url}.")
        
        response = client.chat.completions.create(
            model=service_config["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        simplified = response.choices[0].message.content.strip()
        print(f"Simplified text: '{simplified}'")
        return simplified

    except Exception as e:
        print(f"An error occurred during prompt simplification: {e}")
        # Fallback for cases where the model fails or parsing is difficult
        match = re.search(r"'(.*?)'", prompt)
        if match:
            print("Used regex fallback for simplification.")
            return match.group(1)
        return None
