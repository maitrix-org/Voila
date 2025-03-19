import os
import argparse
import random
import jsonlines
import soundfile as sf
import json
import copy
import torch
from pathlib import Path
from threading import Thread

import torchaudio
from transformers import AutoTokenizer

from model import VoilaAudioAlphaModel, VoilaModel, VoilaAutonomousModel
from spkr import SpeakerEmbedding
from voila_tokenizer import VoilaTokenizer
from tokenize_func import (
    voila_input_format,
    AUDIO_TOKEN_FORMAT,
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_ASSISTANT_TOKEN,
)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_model(model_name, audio_tokenizer_path):
    disable_torch_init()

    if "Voila-audio" in model_name:
        model_type = "audio"
        cls = VoilaAudioAlphaModel
    elif "Voila-auto" in model_name:
        model_type = "autonomous"
        cls = VoilaAutonomousModel
    else:
        model_type = "token"
        cls = VoilaModel

    model = cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        use_cache=True,
    )
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_voila = VoilaTokenizer(model_path=audio_tokenizer_path, device="cuda")
    return model, tokenizer, tokenizer_voila, model_type

def is_audio_output_task(task_type):
    return task_type.endswith("ao") or "aiao" in task_type or "tts" in task_type

def eval_model(model, tokenizer, tokenizer_voila, model_type, task_type, history, ref_embs, ref_embs_mask, max_new_tokens=512):
    # step1: initializing
    num_codebooks = model.config.num_codebooks
    codebook_size = model.config.codebook_size

    AUDIO_MIN_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(0))
    assert isinstance(AUDIO_MIN_TOKEN_ID, int)
    AUDIO_MAX_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(codebook_size*num_codebooks-1))
    assert isinstance(AUDIO_MAX_TOKEN_ID, int)
    AUDIO_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_TOKEN)
    assert isinstance(AUDIO_TOKEN_ID, int)
    ASSISTANT_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_ASSISTANT_TOKEN)
    assert isinstance(ASSISTANT_TOKEN_ID, int)

    # step2: set infer config
    data_cfg = {
        "input_type": model_type,
        "task_type": task_type,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
    }

    # step3: infer
    input_ids, audio_datas, audio_data_masks, streaming_user_input_audio_tokens = voila_input_format(history, tokenizer, tokenizer_voila, data_cfg)

    # prepare user_streaming_generator to simulate streaming user input
    def get_input_generator(all_tokens):
        assert all_tokens is not None
        for i in range(len(all_tokens[0])):
            yield all_tokens[:,i]

    if model_type == "autonomous":
        input_generator = get_input_generator(torch.as_tensor(streaming_user_input_audio_tokens).cuda())
        input_ids = [torch.as_tensor([input]).transpose(1,2).cuda() for input in input_ids]            # transpose to [bs, seq, num_codebooks]
        input_ids = torch.cat(input_ids, dim=2)            # concat to [bs, seq, num_codebooks*2]
    else:
        input_ids = torch.as_tensor([input_ids]).transpose(1,2).cuda()      # transpose to [bs, seq, num_codebooks]
    gen_params = {
        "input_ids": input_ids,
        "ref_embs": ref_embs,
        "ref_embs_mask": ref_embs_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "llm_audio_token_id": AUDIO_TOKEN_ID,
        "min_audio_token_id": AUDIO_MIN_TOKEN_ID,
        "temperature": 0.2,
        "top_k": 50,
        "audio_temperature": 0.8,
        "audio_top_k": 50,
    }
    if model_type == "audio":
        audio_datas = torch.tensor([audio_datas], dtype=torch.bfloat16).cuda()
        audio_data_masks = torch.tensor([audio_data_masks]).cuda()
        gen_params["audio_datas"] = audio_datas
        gen_params["audio_data_masks"] = audio_data_masks
    elif model_type == "autonomous":
        gen_params["input_generator"] = input_generator
        gen_params["llm_assistant_token_id"] = ASSISTANT_TOKEN_ID
    print(f"Input str: {tokenizer.decode(input_ids[0, :, 0])}")
    with torch.inference_mode():
        outputs = model.run_generate(**gen_params)

        if model_type == "autonomous":
            outputs = outputs.chunk(2, dim=2)[1]
        outputs = outputs[0].cpu().tolist()

        predict_outputs = outputs[input_ids.shape[1]:]
        text_outputs = []
        audio_outputs = []
        for _ in range(num_codebooks):
            audio_outputs.append([])
        for item in predict_outputs:
            if item[0] >= AUDIO_MIN_TOKEN_ID and item[0] <= AUDIO_MAX_TOKEN_ID:
                for n, at in enumerate(item):
                    audio_outputs[n].append((at - AUDIO_MIN_TOKEN_ID)%codebook_size)
            elif item[0] != tokenizer.eos_token_id:
                text_outputs.append(item[0])

        out ={
            'text': tokenizer.decode(text_outputs),
        }
        if is_audio_output_task(task_type):
            audio_values = tokenizer_voila.decode(torch.tensor(audio_outputs).cuda())
            out['audio'] = (audio_values.detach().cpu().numpy(), 16000)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--input-text", type=str, default=None)
    parser.add_argument("--input-audio", type=str, default=None)
    parser.add_argument("--result-path", type=str, default="output")
    parser.add_argument("--ref-audio", type=str, default="examples/test1.mp3")
    parser.add_argument("--model-name", type=str, default="maitrix-org/Voila-chat")
    parser.add_argument("--audio-tokenizer-path", type=str, default="maitrix-org/Voila-Tokenizer")
    parser.add_argument("--task-type", type=str, default="chat_aiao")
    args = parser.parse_args()

    assert args.model_name in [
        "maitrix-org/Voila-audio-alpha",
        "maitrix-org/Voila-base",
        "maitrix-org/Voila-chat",
        "maitrix-org/Voila-autonomous-preview",
    ]

    # step0: Model loading
    model, tokenizer, tokenizer_voila, model_type = load_model(args.model_name, args.audio_tokenizer_path)

    # step1: prepare inputs
    Path(args.result_path).mkdir(exist_ok=True, parents=True)
    history = {
        "instruction": args.instruction,
        "conversations": [],
    }
    if args.input_text is not None:
        history["conversations"].append({"from": "user", "text": args.input_text})
    elif args.input_audio is not None:
        history["conversations"].append({"from": "user", "audio": {"file": args.input_audio}})
    else:
        raise Exception("Please provide atleast one of --input-text and --input-audio")
    history["conversations"].append({"from": "assistant"})

    # step2: encode ref
    ref_embs, ref_embs_mask = None, None
    if is_audio_output_task(args.task_type):
        spkr_model = SpeakerEmbedding(device="cuda")
        wav, sr = torchaudio.load(args.ref_audio)
        ref_embs = spkr_model(wav, sr)
        ref_embs_mask = torch.tensor([1]).cuda()

    out = eval_model(model, tokenizer, tokenizer_voila, model_type, args.task_type, history, ref_embs, ref_embs_mask)
    print(f"Output str: {out['text']}")
    if 'audio' in out:
        wav, sr = out['audio']
        save_name = f"{args.result_path}/out.wav"
        sf.write(save_name, wav, sr)
