import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, AutoModelForSpeechSeq2Seq
import os, json
from tqdm import tqdm
import argparse

def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True,)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path,use_safetensors=True,trust_remote_code=True, device_map="auto", cache_dir=".cache")
    return model, processor

def response_to_audio(audio_url, text, model=None, processor=None, temperature = 0.1, max_new_tokens = 512):
    if text == None or text == "":
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_url},
            ]},]
    elif audio_url == None:
        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": text},
           ]},]
    else:
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_url},
                {"type": "text", "text": text},
           ]},]
    
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
    inputs.input_ids = inputs.input_ids.to(model.device)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 
    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature = temperature, repetition_penalty=1.1,  do_sample=True if temperature > 0 else False)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def load_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model and data evaluation")
    # Add arguments
    parser.add_argument('--model_path', type=str, default="SeaLLMs/SeaLLMs-Audio-7B", help='Path to the model')
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

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, model_name + '.json')
    with open(output_path, 'w') as f:
        json.dump(data_eval, f, indent=4, ensure_ascii=False)
    


