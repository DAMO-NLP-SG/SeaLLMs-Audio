from datasets import load_dataset, load_from_disk
import torchaudio
import os

ds = load_dataset("parquet", data_files={"test": "https://huggingface.co/datasets/SeaLLMs/SeaBench-Audio/resolve/main/test.parquet"}, split="test")
# ds = load_from_disk("datasets")
# print(ds)
# print(ds[0])

def process_example(example):
    sample = example["audio"].get_all_samples()
    audio_url = f"data/audio_data/{example['language']}/{example['task']}_{example['idx']}.wav"
    os.makedirs(os.path.dirname(audio_url), exist_ok=True)
    torchaudio.save(audio_url, sample.data, sample.sample_rate)
    example['audio_url'] = audio_url
    return example

ds = ds.map(process_example)
ds = ds.remove_columns(["audio"])

# save the dataset as json
df = ds.to_pandas() 
df.to_json("data/seabench_audio.json", orient="records", indent=4, force_ascii=False)

