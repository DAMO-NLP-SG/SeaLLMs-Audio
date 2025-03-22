# How to train?
SeaLLMs-Audio-7B is based on the [Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B) audio encoder and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) models. To train your own model, you can follow the steps below:
1. Install the required packages by running `pip install -r scripts/requirements.txt`.
2. Prepare the training data. You can refer to the sample data in [`scripts/train_examples.json`](scripts/train_examples.json) for the data format.
3. Initialize the model by running `python scripts/initialize_new_model.py`.
4. Train the model by running `source scripts/sft.sh`.