from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import os

model = Qwen2AudioForConditionalGeneration.from_pretrained("SeaLLMs/SeaLLMs-Audio-7B", device_map="auto")
processor = AutoProcessor.from_pretrained("SeaLLMs/SeaLLMs-Audio-7B")

def response_to_audio(conversation, model=None, processor=None):
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
    if audios != []:
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True,sampling_rate=16000)
    else: 
        inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items() if v is not None}
    generate_ids = model.generate(**inputs, max_new_tokens=2048, temperature = 0, do_sample=False)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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

response = response_to_audio(conversation, model=model, processor=processor)
print(response)

# Audio Analysis
os.system(f"wget -O ASR_en.wav https://damo-nlp-sg.github.io/SeaLLMs-Audio/static/audios/ASR_en.wav")
conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "ASR_en.wav"},
        {"type": "text", "text": "Please write down what is spoken in the audio file."},
    ]},
]

response = response_to_audio(conversation, model=model, processor=processor)
print(response)