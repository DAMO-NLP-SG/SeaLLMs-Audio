from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers import AutoConfig, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    return model, processor

def copy_parameters(model1, model2):
    # copy parameters from model1 to model2
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if param1.shape == param2.shape:
                param2.copy_(param1)
                # print(f"Copied weights from {name1} to {name2}")
            else:
                print(f"Skipped {name1} -> {name2} due to size mismatch")

def initialize_new_model(path_audio, path_25, new_model_path):
    # load the configuration of the models
    list_config_to_update = ['hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'use_sliding_window', 'sliding_window', 'max_window_layers', 'num_key_value_heads', 'hidden_act', 'initializer_range', 'use_cache', 'rope_scaling', 'attention_dropout', 'return_dict', 'output_hidden_states', 'output_attentions', 'torchscript', 'torch_dtype', 'use_bfloat16', 'tf_legacy_loss', 'pruned_heads', 'tie_word_embeddings', 'chunk_size_feed_forward', 'is_encoder_decoder', 'is_decoder', 'cross_attention_hidden_size', 'add_cross_attention', 'tie_encoder_decoder', 'max_length', 'min_length', 'do_sample', 'early_stopping', 'num_beams', 'num_beam_groups', 'diversity_penalty', 'temperature', 'top_k', 'top_p', 'typical_p', 'repetition_penalty', 'length_penalty', 'no_repeat_ngram_size', 'encoder_no_repeat_ngram_size', 'bad_words_ids', 'num_return_sequences', 'output_scores', 'return_dict_in_generate', 'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values', 'exponential_decay_length_penalty', 'suppress_tokens', 'begin_suppress_tokens', 'finetuning_task', 'id2label', 'label2id', 'tokenizer_class', 'prefix', 'bos_token_id', 'pad_token_id', 'eos_token_id', 'sep_token_id', 'decoder_start_token_id', 'task_specific_params', 'problem_type', 'transformers_version', 'model_type']
    config_audio = AutoConfig.from_pretrained(path_audio)
    config_25 = AutoConfig.from_pretrained(path_25)
    generation_config = GenerationConfig.from_pretrained(path_audio)

    for k in list_config_to_update:
        setattr(config_audio.text_config, k, config_25.to_dict()[k])

    config_audio.save_pretrained(new_model_path)
    generation_config.save_pretrained(new_model_path)

    # merge the weights of LLM and audio encoder
    model_audio, processor = load_model(path_audio)
    model_25 = AutoModelForCausalLM.from_pretrained(
        path_25,
        torch_dtype="auto",
        device_map="auto"
    )

    # create the new model
    model = Qwen2AudioForConditionalGeneration(config=config_audio)

    # copy the weights
    vocab_size = model_25.model.embed_tokens.weight.data.size()[0]
    copy_parameters(model_25, model.language_model)
    copy_parameters(model_audio.audio_tower, model.audio_tower)
    model.language_model.model.embed_tokens.weight.data[:vocab_size,].copy_(model_25.model.embed_tokens.weight.data)
    model.language_model.lm_head.weight.data[:vocab_size,].copy_(model_25.lm_head.weight.data)

    model.save_pretrained(new_model_path)
    processor.save_pretrained(new_model_path)

if __name__ == "__main__":
    path_audio = "Qwen/Qwen2-Audio-7B" # path to the audio model
    path_25 = "Qwen/Qwen2.5-7B-Instruct" # path to the LLM model
    new_model_path = "checkpoints/Qwen2.5-7B-Instruct-Audio-base"
    initialize_new_model(path_audio, path_25, new_model_path)


    