import os
import random
import shutil
import pickle
import gradio as gr
import soundfile as sf
from pathlib import Path

import torch
import torchaudio

from huggingface_hub import hf_hub_download

from infer import load_model, eval_model
from spkr import SpeakerEmbedding


spkr_model = SpeakerEmbedding(device="cuda")
model, tokenizer, tokenizer_voila, model_type = load_model("maitrix-org/Voila-chat", "maitrix-org/Voila-Tokenizer")
default_ref_file = "examples/character_ref_emb_demo.pkl"
default_ref_name = "Homer Simpson"
million_voice_ref_file = hf_hub_download(repo_id="maitrix-org/Voila-million-voice", filename="character_ref_emb_chunk0.pkl", repo_type="dataset")

instruction = "You are a smart AI agent created by Maitrix.org."
save_path = "output"

intro = """**Voila**

For more demos, please goto [https://voila.maitrix.org](https://voila.maitrix.org)."""

default_ref_emb_mask_list = pickle.load(open(default_ref_file, "rb"))
million_voice_ref_emb_mask_list = pickle.load(open(million_voice_ref_file, "rb"))

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

def call_bot(history, ref_embs, request: gr.Request, progress=gr.Progress()):
    progress(0, desc="Processing your message...")
    formated_history = {
        "instruction": instruction,
        "conversations": [{'from': item["role"], 'audio': {"file": item["content"][0]}} for item in history],
    }
    formated_history["conversations"].append({"from": "assistant"})
    print(formated_history)
    ref_embs = torch.tensor(ref_embs, dtype=torch.float32, device="cuda")
    ref_embs_mask = torch.tensor([1], device="cuda")
    progress(0.3, desc="Generating response...")
    out = eval_model(model, tokenizer, tokenizer_voila, model_type, "chat_aiao", formated_history, ref_embs, ref_embs_mask, max_new_tokens=512)
    if 'audio' in out:
        progress(0.7, desc="Processing audio...")
        wav, sr = out['audio']

        # Ensure output directory exists
        Path(save_path).mkdir(exist_ok=True, parents=True)
        user_dir = Path(f"{save_path}/{str(request.session_hash)}")
        user_dir.mkdir(exist_ok=True, parents=True)
        save_name = f"{user_dir}/{len(history)}.wav"
        sf.write(save_name, wav, sr)

        history.append({"role": "assistant", "content": {"path": save_name}})
    else:
        history.append({"role": "assistant", "content": {"text": out['text']}})
    progress(1.0, desc="Complete!")

    return history

def run_tts(text, ref_embs, progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Please enter some text to convert to speech")
    try:
        progress(0, desc="Processing text...")
        formated_history = {
            "instruction": "",
            "conversations": [{'from': "user", 'text': text}],
        }
        formated_history["conversations"].append({"from": "assistant"})
        ref_embs = torch.tensor(ref_embs, dtype=torch.float32, device="cuda")
        ref_embs_mask = torch.tensor([1], device="cuda")
        progress(0.5, desc="Generating speech...")
        out = eval_model(model, tokenizer, tokenizer_voila, model_type, "chat_tts", formated_history, ref_embs, ref_embs_mask, max_new_tokens=512)
        if 'audio' in out:
            progress(1.0, desc="Complete!")
            wav, sr = out['audio']
            return (sr, wav)
        else:
            raise Exception("No audio output")
    except Exception as e:
        raise gr.Error(f"Error generating speech: {str(e)}")

def run_asr(audio, progress=gr.Progress()):
    if not audio:
        raise gr.Error("Please provide an audio file or record your voice")
    try:
        progress(0, desc="Processing audio...")
        formated_history = {
            "instruction": "",
            "conversations": [{'from': "user", 'audio': {"file": audio}}],
        }
        formated_history["conversations"].append({"from": "assistant"})
        progress(0.5, desc="Converting to text...")
        out = eval_model(model, tokenizer, tokenizer_voila, model_type, "chat_asr", formated_history, None, None, max_new_tokens=512)
        if 'text' in out:
            progress(1.0, desc="Complete!")
            return out['text']
        else:
            raise Exception("No text output")
    except Exception as e:
        raise gr.Error(f"Error converting speech to text: {str(e)}")


def markdown_ref_name(ref_name):
    return f"### Current voice id: {ref_name}"

def random_million_voice():
    voice_id = random.choice(list(million_voice_ref_emb_mask_list.keys()))
    return markdown_ref_name(voice_id), million_voice_ref_emb_mask_list[voice_id]

def get_ref_modules(cur_ref_embs):
    with gr.Row() as ref_row:
        with gr.Group(elem_id="voice_selector_box"):
            with gr.Row():
                current_ref_name = gr.Markdown(
                    markdown_ref_name(default_ref_name),
                    elem_id="current_voice"
                )
            with gr.Row() as ref_name_row:
                with gr.Column(scale=2):
                    ref_name_dropdown = gr.Dropdown(
                        choices=list(default_ref_emb_mask_list.keys()),
                        value=default_ref_name,
                        label="Select Voice",
                        min_width=160,
                        elem_id="voice_selector"
                    )
                with gr.Column(scale=1):
                    random_ref_button = gr.Button(
                        "🎲 Random Voice",
                        size="md",
                        variant="secondary",
                        elem_id="random_voice"
                    )
            with gr.Row(visible=False) as ref_audio_row:
                with gr.Column(scale=2):
                    ref_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        show_label=False,
                        min_width=80,
                        elem_id="ref_audio_input"
                    )
                with gr.Column(scale=1):
                    change_ref_button = gr.Button(
                        "Change Voice",
                        interactive=False,
                        min_width=80,
                        variant="primary",
                        elem_id="change_voice_button"
                    )
    ref_name_dropdown.change(
        lambda x: (markdown_ref_name(x), default_ref_emb_mask_list[x]),
        ref_name_dropdown,
        [current_ref_name, cur_ref_embs]
    )
    random_ref_button.click(
        random_million_voice,
        None,
        [current_ref_name, cur_ref_embs],
    )
    ref_audio.input(lambda: gr.Button(interactive=True), None, change_ref_button)
    custom_ref_voice = gr.Checkbox(label="Use custom voice", value=False)
    def custom_ref_voice_change(x, cur_ref_embs, cur_ref_embs_mask):
        if not x:
            cur_ref_embs = default_ref_emb_mask_list[default_ref_name]
        return [gr.Row(visible=not x), gr.Audio(value=None), gr.Row(visible=x), markdown_ref_name("Custom voice"), cur_ref_embs]
    custom_ref_voice.change(
        custom_ref_voice_change,
        [custom_ref_voice, cur_ref_embs],
        [ref_name_row, ref_audio, ref_audio_row, current_ref_name, cur_ref_embs]
    )
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
            with gr.Group(elem_id="voice_input_box"):
                with gr.Row():
                    with gr.Column(scale=4):
                        chat_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            show_label=False,
                            elem_id="voice_input",
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        submit = gr.Button(
                            "Send",
                            variant="primary",
                            size="lg",
                            interactive=False,
                            elem_id="send_button"
                        )
            gr.Markdown(intro)
        with gr.Column(scale=9):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                bubble_full_width=False,
                scale=1,
                show_copy_button=True,
                height=600,
                container=True,
                avatar_images=(
                    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/lion.jpg",
                    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/lion.jpg",
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
            with gr.Group():
                tts_output = gr.Audio(
                    label="Generated Speech",
                    interactive=False,
                    elem_id="tts_output"
                )
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text to Speech",
                        placeholder="Enter text to convert to speech...",
                        info="Enter the text you want to convert to speech",
                        elem_id="tts_input",
                        lines=5
                    )
                    submit = gr.Button(
                        "Generate Speech",
                        variant="primary",
                        size="lg",
                        elem_id="tts_submit"
                    )
        submit.click(
            run_tts, [text_input, cur_ref_embs], tts_output
        )
    return tts_tab

def get_asr_tab():
    with gr.Row() as asr_tab:
        with gr.Column():
            with gr.Group():
                asr_input = gr.Audio(
                    label="ASR input",
                    sources=["microphone", "upload"],
                    type="filepath",
                    elem_id="asr_input"
                )
                submit = gr.Button(
                    "Convert to Text",
                    variant="primary",
                    size="lg",
                    elem_id="asr_submit"
                )
            gr.Markdown(intro)
        with gr.Column():
            asr_output = gr.Textbox(
                label="Transcribed Text",
                interactive=False,
                elem_id="asr_output",
                lines=5
            )
    submit.click(
        run_asr, [asr_input], asr_output
    )
    return asr_tab

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size="md",
        text_size="md",
        font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    ),
    fill_height=True,
    css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .chat-message {
            padding: 1.5rem !important;
            border-radius: 0.5rem !important;
            margin-bottom: 1rem !important;
            display: flex !important;
            flex-direction: row !important;
            align-items: flex-start !important;
            gap: 1rem !important;
        }
        .chat-message.user {
            background-color: #f0f4ff !important;
        }
        .chat-message.assistant {
            background-color: #f8fafc !important;
        }
        .chat-message .avatar {
            width: 40px !important;
            height: 40px !important;
            border-radius: 50% !important;
        }
        .chat-message .message {
            flex: 1 !important;
            padding: 0.5rem !important;
        }
        .gradio-button {
            transition: all 0.2s ease-in-out !important;
        }
        .gradio-button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important;
        }
    """
) as demo:
    gr.Markdown("# 🎙️ Voila - Voice AI Assistant")
    with gr.Tabs() as tabs:
        with gr.Tab("Chat", id="chat_tab"):
            chat_tab = get_chat_tab()
        with gr.Tab("Text to Speech", id="tts_tab"):
            tts_tab = get_tts_tab()
        with gr.Tab("Speech to Text", id="asr_tab"):
            asr_tab = get_asr_tab()
    demo.unload(delete_directory)

if __name__ == "__main__":
    demo.launch(share=True)
