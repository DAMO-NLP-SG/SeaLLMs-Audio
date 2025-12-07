from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import AutoPeftModel, AutoPeftModelForCausalLM
import gradio as gr
import os, json
from sys import argv
from tqdm import tqdm
import argparse

def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True,)
    model =  AutoModelForSpeechSeq2Seq.from_pretrained(model_path,use_safetensors=True,trust_remote_code=True, device_map="auto")
    return model, processor

def response_to_audio(audio_url, text=None, model=None, processor=None, temperature = 0, max_new_tokens = 512):
    prompt = "Given the following audio context: <SpeechHere>\n\nText instruction: {query}"
    query = "Please follow the instruction in the speech." if text == None else text
    conversation = [
        {"role": "user", "content": prompt.format(query=query)}
    ]

    chat_prompt = processor.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    array = librosa.load(audio_url, sr=processor.feature_extractor.sampling_rate)[0]

    inputs = processor(text=chat_prompt, audios=array)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample = temperature > 0 , temperature=temperature, repetition_penalty=1.1) #top_p=0.9, 
    generated_ids = outputs[:, inputs['input_ids'].size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def load_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model and data evaluation")
    # Add arguments
    parser.add_argument('--model_path', type=str, default="MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION", help='Path to the model')
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
    


