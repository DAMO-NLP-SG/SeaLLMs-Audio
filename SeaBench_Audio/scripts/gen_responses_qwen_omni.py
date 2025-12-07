from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, AutoModelForSpeechSeq2Seq, Qwen2_5OmniForConditionalGeneration
from peft import AutoPeftModel, AutoPeftModelForCausalLM
import gradio as gr
import os, json
from sys import argv
from tqdm import tqdm
import argparse
import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def load_model(model_path):
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path,trust_remote_code=True,)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    return model, processor

def response_to_audio(audio_url, text=None, model=None, processor=None, temperature = 0, max_new_tokens = 512):
    conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_url},
            # {"type": "text", "text": "Please transcribe this speech."},
        ],
    },
    ]

    if text != None:
        conversation[-1]['content'].append({"type": "text", "text": text})

    # set use audio in video
    USE_AUDIO_IN_VIDEO = False
    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids, audio = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample = temperature > 0 , temperature=temperature, repetition_penalty=1.1) #top_p=0.9, 
    response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("\nassistant\n")[-1]

    return response

def load_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model and data evaluation")
    # Add arguments
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-Omni-7B", help='Path to the model')
    parser.add_argument('--data_path', type=str, nargs='?', default="data/seabench_audio.json", help='Path to the data (optional)')
    parser.add_argument('--output_dir', type=str, default="responses", nargs='?', help='Path to the output file (optional)')
    # Parse arguments
    args = parser.parse_args()

    # Data path
    data_path = args.data_path
    data_name = data_path.split('/')[-1].split('.')[-2]
    data_eval = load_json(data_path)

    # Process model path
    model_path = args.model_path
    model_path = model_path[:-1] if model_path[-1] == '/' else model_path
    model_name = model_path.split('/')[-1]
    model, processor = load_model(model_path)

    func_respond = response_to_audio
    for d in tqdm(data_eval):
        audio_url = d['audio_url']
        text = d['query']
        prediction = func_respond(audio_url, text, model, processor, temperature = 0, max_new_tokens = 2048)
        d[model_name] = prediction

    output_path = os.path.join(args.output_dir, model_name + '.json')
    with open(output_path, 'w') as f:
        json.dump(data_eval, f, indent=4, ensure_ascii=False)
    


