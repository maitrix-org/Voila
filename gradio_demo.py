import os
import random
import shutil
import pickle
import gradio as gr
import soundfile as sf
from pathlib import Path

import torch
import torchaudio

from infer import load_model, eval_model
from spkr import SpeakerEmbedding


spkr_model = SpeakerEmbedding(device="cuda")
model, tokenizer, tokenizer_voila, model_type = load_model("maitrix-org/Voila-chat", "maitrix-org/Voila-Tokenizer")
default_ref_file = "examples/character_ref_emb_demo.pkl"
default_ref_name = "Homer Simpson"

instruction = "You are a smart AI agent created by Maitrix.org."
save_path = "output"

intro = """**Voila**

For more demos, please goto [https://voila.maitrix.org](https://voila.maitrix.org)."""

default_ref_emb_mask_list = pickle.load(open(default_ref_file, "rb"))

def get_ref_embs(ref_audio):
    wav, sr = torchaudio.load(ref_audio)
    ref_embs = spkr_model(wav, sr).cpu()
    return ref_embs

def delete_directory(request: gr.Request):
    if not request.session_hash:
        return
    user_dir = Path(f"{save_path}/{str(request.session_hash)}")
    if user_dir.exists():
        shutil.rmtree(str(user_dir))

def add_message(history, message):
    history.append({"role": "user", "content": {"path": message}})
    return history, gr.Audio(value=None), gr.Button(interactive=False)

def call_bot(history, ref_embs, request: gr.Request):
    formated_history = {
        "instruction": instruction,
        "conversations": [{'from': item["role"], 'audio': {"file": item["content"][0]}} for item in history],
    }
    formated_history["conversations"].append({"from": "assistant"})
    print(formated_history)
    ref_embs = torch.tensor(ref_embs, dtype=torch.float32, device="cuda")
    ref_embs_mask = torch.tensor([1], device="cuda")
    out = eval_model(model, tokenizer, tokenizer_voila, model_type, "chat_aiao", formated_history, ref_embs, ref_embs_mask, max_new_tokens=512)
    if 'audio' in out:
        wav, sr = out['audio']

        user_dir = Path(f"{save_path}/{str(request.session_hash)}")
        user_dir.mkdir(exist_ok=True)
        save_name = f"{user_dir}/{len(history)}.wav"
        sf.write(save_name, wav, sr)

        history.append({"role": "assistant", "content": {"path": save_name}})
    else:
        history.append({"role": "assistant", "content": {"text": out['text']}})

    return history

def run_tts(text, ref_embs):
    formated_history = {
        "instruction": "",
        "conversations": [{'from': "user", 'text': text}],
    }
    formated_history["conversations"].append({"from": "assistant"})
    ref_embs = torch.tensor(ref_embs, dtype=torch.float32, device="cuda")
    ref_embs_mask = torch.tensor([1], device="cuda")
    out = eval_model(model, tokenizer, tokenizer_voila, model_type, "chat_tts", formated_history, ref_embs, ref_embs_mask, max_new_tokens=512)
    if 'audio' in out:
        wav, sr = out['audio']
        return (sr, wav)
    else:
        raise Exception("No audio output")

def run_asr(audio):
    formated_history = {
        "instruction": "",
        "conversations": [{'from': "user", 'audio': {"file": audio}}],
    }
    formated_history["conversations"].append({"from": "assistant"})
    out = eval_model(model, tokenizer, tokenizer_voila, model_type, "chat_asr", formated_history, None, None, max_new_tokens=512)
    if 'text' in out:
        return out['text']
    else:
        raise Exception("No text output")

def get_ref_modules(cur_ref_embs):
    with gr.Row() as ref_row:
        with gr.Row() as ref_name_row:
            ref_name_dropdown = gr.Dropdown(
                choices=list(default_ref_emb_mask_list.keys()),
                value=default_ref_name,
                label="Reference voice",
                min_width=160,
            )
        with gr.Row(visible=False) as ref_audio_row:
            with gr.Column(scale=2, min_width=80):
                ref_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    show_label=False,
                    min_width=80,
                )
            with gr.Column(scale=1, min_width=80):
                change_ref_button = gr.Button(
                    "Change voice",
                    interactive=False,
                    min_width=80,
                )
    ref_name_dropdown.change(
        lambda x: default_ref_emb_mask_list[x],
        ref_name_dropdown,
        cur_ref_embs
    )
    ref_audio.input(lambda: gr.Button(interactive=True), None, change_ref_button)
    # If custom ref voice checkbox is checked, show the Audio component to record or upload a reference voice
    custom_ref_voice = gr.Checkbox(label="Use custom voice", value=False)
    # Checked: enable audio and button
    # Unchecked: disable audio and button
    def custom_ref_voice_change(x, cur_ref_embs, cur_ref_embs_mask):
        if not x:
            cur_ref_embs = default_ref_emb_mask_list[default_ref_name]
        return [gr.Row(visible=not x), gr.Audio(value=None), gr.Row(visible=x), cur_ref_embs]
    custom_ref_voice.change(
        custom_ref_voice_change,
        [custom_ref_voice, cur_ref_embs],
        [ref_name_row, ref_audio, ref_audio_row, cur_ref_embs]
    )
    # When change ref button is clicked, get the reference voice and update the reference voice state
    change_ref_button.click(
        lambda: gr.Button(interactive=False), None, [change_ref_button]
    ).then(
        get_ref_embs, ref_audio, cur_ref_embs
    )
    return ref_row

def get_chat_tab():
    cur_ref_embs = gr.State(default_ref_emb_mask_list[default_ref_name])
    with gr.Row() as chat_tab:
        with gr.Column(scale=1):
            ref_row = get_ref_modules(cur_ref_embs)
            # Voice chat input
            chat_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                show_label=False,
            )
            submit = gr.Button("Submit", interactive=False)
            gr.Markdown(intro)
        with gr.Column(scale=9):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                bubble_full_width=False,
                scale=1,
                show_copy_button=False,
                avatar_images=(
                    None,  # os.path.join("files", "avatar.png"),
                    None, # os.path.join("files", "avatar.png"),
                ),
            )

        chat_input.input(lambda: gr.Button(interactive=True), None, submit)
        submit.click(
            add_message, [chatbot, chat_input], [chatbot, chat_input, submit]
        ).then(
            call_bot, [chatbot, cur_ref_embs], chatbot, api_name="bot_response"
        )
    return chat_tab

def get_tts_tab():
    cur_ref_embs = gr.State(default_ref_emb_mask_list[default_ref_name])
    with gr.Row() as tts_tab:
        with gr.Column(scale=1):
            ref_row = get_ref_modules(cur_ref_embs)
            gr.Markdown(intro)
        with gr.Column(scale=9):
            tts_output = gr.Audio(label="TTS output", interactive=False)
            with gr.Row():
                text_input = gr.Textbox(label="Text", placeholder="Text to TTS")
                submit = gr.Button("Submit")
        submit.click(
            run_tts, [text_input, cur_ref_embs], tts_output
        )
    return tts_tab

def get_asr_tab():
    with gr.Row() as asr_tab:
        with gr.Column():
            asr_input = gr.Audio(
                label="ASR input",
                sources=["microphone", "upload"],
                type="filepath",
            )
            submit = gr.Button("Submit")
            gr.Markdown(intro)
        with gr.Column():
            asr_output = gr.Textbox(label="ASR output", interactive=False)
    submit.click(
        run_asr, [asr_input], asr_output
    )
    return asr_tab

with gr.Blocks(fill_height=True) as demo:
    with gr.Tab("Chat"):
        chat_tab = get_chat_tab()
    with gr.Tab("TTS"):
        tts_tab = get_tts_tab()
    with gr.Tab("ASR"):
        asr_tab = get_asr_tab()
    demo.unload(delete_directory)

if __name__ == "__main__":
    demo.launch(share=True)
