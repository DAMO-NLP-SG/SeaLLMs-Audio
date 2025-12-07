import os, time
from google import genai
from google.genai import types

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random
)

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(6), before=before_retry_fn)
def generate_response(audio_path, prompt=None, model_name="gemini-2.5-flash", temperature=0, api_key = None):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY") if api_key is None else api_key,
    )

    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
            # include_thoughts=True,
        ),
        response_mime_type="text/plain",
        max_output_tokens=8192,
        top_p=0.95,
        top_k=40,
    )

    myfile = client.files.upload(file=audio_path)

    try:
        response = client.models.generate_content(
            model=model_name, contents=[prompt, myfile],
            config=generate_content_config,
        )
        return response.text
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Caution: Failed to generate response."


def parallel_generate_response(args):
    return generate_response(*args)

if __name__ == "__main__":
    audio_path = "data/audio_data/en/ASR_0.wav"
    prompt = "Please transcribe the audio."
    model_name = "gemini-2.5-flash"
    response = generate_response(audio_path, prompt, model_name)
    print(response)

