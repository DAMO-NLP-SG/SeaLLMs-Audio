import json
from utils import generate_response, parallel_generate_response
from tqdm import tqdm
import concurrent.futures
import random
import re
import argparse
import os

random.seed(42)

template = """Please act as an impartial judge and evaluate the quality of the text response provided by an AI assistant to the user question. The user question may be in text or in audio form. An audio .wav file for the question is also given, which the AI assistant must analyze in order to give a correct response. Begin your evaluation process by first analyzing the content of the corresponding audio file, and comparing the assistant's answer against the reference answer. The audio content is a key component of the evaluation and please use it to identify any inaccuracies, contextual misunderstandings, and language choice issues in the assistant's response. An Evaluation Scoring rubric is provided alongside each assistant's answer and must be strictly adhered to, with the ratings assigned on a sequential, first-match basis. Be as objective as possible and your explanation should not exceed two paragraphs. After providing your explanation in English, you must rate the response on a scale of 1 to 5 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[Start of Reference Answer]
{reference}
[End of Reference Answer]

[Start of Assistant's Answer]
{answer}
[End of Assistant's Answer]

[Start of Evaluation Scoring Guide]
{rule}
[End of Evaluation Scoring Guide]"""

# read rules.json
with open("data/rules.json", "r") as f:
    rules = json.load(f)

def extract_score(response):
    score = re.search(r'Rating: \[\[(\d+)\]', response)
    if score:
        return int(score.group(1))
    else:
        return None

def llm_as_judge(model, judgement_model, input_folder, output_folder, max_workers, languages, tasks, indexs, api_key=None):
    file_path = f"{input_folder}/{model}.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    if languages is not None:
        languages = languages.split(',')
        data = [d for d in data if d['language'] in languages]
    if tasks is not None:
        tasks = tasks.split(',')
        data = [d for d in data if d['task'] in tasks]
    if indexs is not None:
        indexs = indexs.split(',')
        data = [d for d in data if str(d['idx']) in indexs]


    print(f"Evaluating {len(data)} questions")

    prompt_args = []
    for d in data:
        formatted_prompt = template.format(question=d["query"], reference=d["reference"], answer=d[model], rule=rules[d["task"].split("_")[0]])
        d['judgement_prompt'] = formatted_prompt
        prompt_args.append((d["audio_url"], formatted_prompt, judgement_model, 0, api_key))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        responses = list(tqdm(executor.map(parallel_generate_response, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))

    for d, response in zip(data, responses):
        d['eval'] = response
        d['rating'] = extract_score(response)

    os.makedirs(output_folder, exist_ok=True)
    output_file_path = f"{output_folder}/{judgement_model.split('/')[0]}_eval_{model.split('/')[0]}.json"
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def llm_as_judge_post(model, judgement_model, input_folder, output_folder, max_workers, languages, tasks, indexs, api_key=None):

    output_file_path = f"{output_folder}/{judgement_model.split('/')[0]}_eval_{model.split('/')[0]}.json"
    with open(output_file_path, "r") as f:
        data = json.load(f)

    print(f"Evaluating {len(data)} questions")

    for d in data:
        if d['eval'] == "Caution: Failed to generate response." or d['rating'] is None:
            response = generate_response(d["audio_url"], d['judgement_prompt'], judgement_model, 0.5, api_key)
            d['eval'] = response
            d['rating'] = extract_score(response)
            print(f"Rating for {d['language']} {d['task']} {d['idx']} is {d['rating']}")
        else:
            continue

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Run LLM as judge for audio/text QA evaluation.")
    parser.add_argument('--model', type=str, default="SeaLLMs-Audio-7B", help='Name of the model to evaluate')
    parser.add_argument('--judgement_model', type=str, default="gemini-2.5-flash", help='Name of the judgement model')
    parser.add_argument('--input_folder', type=str, default="responses", help='Input folder containing model outputs')
    parser.add_argument('--output_folder', type=str, default="judgements", help='Output folder for judgements')
    parser.add_argument('--max_workers', type=int, default=5, help='Maximum number of workers')
    parser.add_argument('--languages', type=str, default=None, help='Languages to evaluate')
    parser.add_argument('--tasks', type=str, default=None, help='Tasks to evaluate')
    parser.add_argument('--indexs', type=str, default=None, help='Indexs to evaluate')
    parser.add_argument('--api_key', type=str, default=None, help='API key for the judgement model')
    parser.add_argument('--post', action='store_true', help='Whether to post-process the judgements')
    args = parser.parse_args()

    if args.post:
        llm_as_judge_post(args.model, args.judgement_model, args.input_folder, args.output_folder, args.max_workers, args.languages, args.tasks, args.indexs, args.api_key)
    else:
        llm_as_judge(args.model, args.judgement_model, args.input_folder, args.output_folder, args.max_workers, args.languages, args.tasks, args.indexs, args.api_key)

if __name__ == "__main__":
    main()

