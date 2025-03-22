from vllm import LLM, SamplingParams
import librosa, os
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("SeaLLMs/SeaLLMs-Audio-7B")
llm = LLM(
    model="SeaLLMs/SeaLLMs-Audio-7B", trust_remote_code=True, gpu_memory_utilization=0.5,  
    enforce_eager=True,  device = "cuda",
    limit_mm_per_prompt={"audio": 5},
)

def response_to_audio(conversation, model=None, processor=None, temperature = 0.1,repetition_penalty=1.1, top_p = 0.9,max_new_tokens = 4096):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if ele['audio_url'] != None:
                        audios.append(librosa.load(
                            ele['audio_url'], 
                            sr=processor.feature_extractor.sampling_rate)[0]
                        )

    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_new_tokens, repetition_penalty=repetition_penalty, top_p=top_p, top_k=20,
        stop_token_ids=[],
    )

    input = {
            'prompt': text,
            'multi_modal_data': {
                'audio': [(audio, 16000) for audio in audios]
            }
            }

    output = model.generate([input], sampling_params=sampling_params)[0]
    response = output.outputs[0].text
    return response

# Voice Chat
os.system(f"wget -O fact_en.wav https://damo-nlp-sg.github.io/SeaLLMs-Audio/static/audios/fact_en.wav")
os.system(f"wget -O general_en.wav https://damo-nlp-sg.github.io/SeaLLMs-Audio/static/audios/general_en.wav")
conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "fact_en.wav"},
    ]},
    {"role": "assistant", "content": "The most abundant gas in Earth's atmosphere is nitrogen. It makes up about 78 percent of the atmosphere by volume."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "general_en.wav"},
    ]},
]

response = response_to_audio(conversation, model=llm, processor=processor)
print(response)

# Audio Analysis
os.system(f"wget -O ASR_en.wav https://damo-nlp-sg.github.io/SeaLLMs-Audio/static/audios/ASR_en.wav")
conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "ASR_en.wav"},
        {"type": "text", "text": "Please write down what is spoken in the audio file."},
    ]},
]

response = response_to_audio(conversation, model=llm, processor=processor)
print(response)