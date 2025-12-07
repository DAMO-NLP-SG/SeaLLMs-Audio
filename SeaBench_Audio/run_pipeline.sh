# prepare data
python prepare_data.py

# Generate model responses
python gen_responses.py --model_path SeaLLMs/SeaLLMs-Audio-7B
# python gen_responses.py --model_path Qwen/Qwen2-Audio-7B-Instruct
# python scripts/gen_responses_MERaLion.py
# python scripts/gen_responses_MERaLion2.py
# python scripts/gen_responses_qwen_omni.py

# # Generate judgements
judgement_model=gemini-2.5-flash
output_folder=judgements
api_key=$GEMINI_API_KEY

# for model in SeaLLMs-Audio-7B Qwen2-Audio-7B-Instruct MERaLiON-AudioLLM-Whisper-SEA-LION MERaLiON-2-10B Qwen2.5-Omni-7B; do
for model in SeaLLMs-Audio-7B; do
    echo "Generating judgements for $model with $judgement_model"
    python gen_judgements.py --model $model \
        --judgement_model $judgement_model \
        --output_folder $output_folder \
        --max_workers 2 \
        --api_key $api_key
done

# Generate summary
judgement_folder=$output_folder
summary_folder=summary
python gen_summary.py --judgement_folder $judgement_folder --summary_folder $summary_folder


