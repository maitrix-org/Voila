<p align="center">
    <img src="https://maitrix-org.github.io/Voila-blog/static/images/logo.png" width="400"/><br/>
    <b>Voila: <span style="color:#ca00f9">Voi</span>ce-<span style="color:#ca00f9">La</span>nguage Foundation Models</b><br/><br/>
    💜 <a href="https://maitrix-org.github.io/Voila-blog"><b>Voila</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/maitrix-org/Voila">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/maitrix-org/voila-67e0d96962c19f221fc73fa5">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper (Coming soon)</a> &nbsp&nbsp | &nbsp&nbsp 🌐 <a href="https://huggingface.co/spaces/maitrix-org/Voila-demo">Demo</a>
</p>

Voila is a groundbreaking family of large audio-language foundation models that revolutionizes human-AI interactions. Breaking away from the constraints of traditional voice AI systems—high latency, loss of vocal nuances, and mechanical responses, Voila employs an innovative end-to-end model design and a novel hierarchical Transformer architecture. This approach enables real-time, autonomous, and rich voice interactions, with latency as low as 195 ms, surpassing average human response times. Combining advanced voice and language modeling, Voila offers customizable, persona-driven engagements and excels in a range of audio tasks from ASR and TTS to speech translation across six languages. With the online [web demo](https://huggingface.co/spaces/maitrix-org/Voila-demo), Voila invites you to explore a transformative, natural dialogue experience between human and AI.

# ✨ Highlights
- ⭐ High-fidelity, low-latency, real-time streaming audio processing
- ⭐ Effective integration of voice and language modeling capabilities
- ⭐ Millions of pre-built and custom voices, fast voice switching during conversation
- ⭐ Unified model for various audio tasks

# 🎥 Video Demo
<div align="center">
    <video width="60%" controls>
        <source src="https://maitrix-org.github.io/Voila-blog/static/videos/voila-demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

# 🔥 Latest News!!

* April 28, 2025: 👋 We've released the inference code and model weights of Voila.

# ⚙️ Foundation Models

| Model | Description | Download Link |
|--------|-----------|-----------------|
|Voila-base|Voila base model|https://huggingface.co/maitrix-org/Voila-base|
|Voila-Chat|End-to-end audio chat model|https://huggingface.co/maitrix-org/Voila-chat|
|Voila-Autonomous (preview)|Full-duplex audio chat model|https://huggingface.co/maitrix-org/Voila-autonomous-preview|
|Voila-Audio-alpha|Empowering LLM with raw audio input|https://huggingface.co/maitrix-org/Voila-audio-alpha|
|Voila-Tokenizer|Audio tokenizer|https://huggingface.co/maitrix-org/Voila-Tokenizer|

## Usage
### CLI demo
```shell
for model_name in "maitrix-org/Voila-audio-alpha" "maitrix-org/Voila-base" "maitrix-org/Voila-chat"; do
    # Text chat
    python infer.py \
        --model-name ${model_name} \
	    --instruction "" \
	    --input-text "Hello" \
	    --task-type chat_tito
    # Voice chat
    python infer.py \
        --model-name ${model_name} \
	    --instruction "" \
	    --input-audio "examples/test1.mp3" \
	    --task-type chat_aiao
done

# Autonomous mode
python infer.py \
    --model-name "maitrix-org/Voila-autonomous-preview" \
	--instruction "" \
	--input-audio "examples/test_autonomous1.mp3" \
	--task-type chat_aiao_auto
```

### Gradio demo
```shell
python gradio_demo.py
```

For more information, please refer to the [code repository](https://github.com/maitrix-org/Voila).

# 📁 Datasets
We publish the following two datasets: Voila Benchmark and Voila Voice Library. Voila-Benchmark is a novel speech evaluation benchmark, while Voila Voice Library provides millions of pre-built and customizable voices.

| Dataset | Description | Download Link |
|--------|-----------|-----------------|
|Voila Benchmark| Evaluation of Voila Benchmark | https://huggingface.co/datasets/maitrix-org/Voila-Benchmark |
|Voila Voice Library| Millons of pre-build voices | https://huggingface.co/datasets/maitrix-org/Voila-million-voice

# 📊 Benchmark
## 1. Voila Benchmark
We introduce a novel speech evaluation benchmark called the VoilaBenchmark. The Voila Benchmark is constructed by sampling from five widely used language model evaluation datasets: MMLU, MATH, OpenAI HumanEval, NQ-Open, and GSM8k. We compare our results with SpeechGPT and Moshi.
| Model | Voila Benchmark |
|-------|----------------|
|SpeechGPT| 13.29|
|Moshi | 11.45 |
|**Voila** | **30.56** |

_(higher is better)_

For detailed scores of Voila Benchmark on each specific domain, please refer to our paper (Section 5.1 "Evaluation of Voila Benchmark").
## 2. Evaluation of ASR
As Voila supports multiple tasks, including Automatic Speech Recognition (ASR), Text-to-Speech(TTS), and spoken question answering, we also evaluate the performance of ASR and TTS. 
For ASR, we assess performance on the LibriSpeech test-clean dataset, using Word Error Rate (WER) as our metric. Voila attains a word error rate (WER) of 4.8%, outperforming the 5.7% reported by Moshi. In scenarios where both models utilize LibriSpeech training data, Voila achieves an impressive WER of 2.7%.
| Model | LibriSpeech test-clean (WER) |
|-------|-----------------------|
|Whisper large v2|2.7|
|Whisper large v3|2.2|
|FastConformer|3.6|
|VoxtLM |2.7|
|Moshi |5.7|
|**Voila (w/o LibriSpeech train split)** |**4.8**|
|**Voila (with LibriSpeech train split)**|**2.7**|

_(lower is better)_

## 3. Evaluation of TTS
For TTS, we follow the evaluation metrics proposed in Vall-E, which involves transcribing the generated audio using HuBERT-Large.
Voila once again leads with a WER of 3.2% (and 2.8% when using LibriSpeech training data).

| Model | LibriSpeech test-clean (WER) |
|-------|-----------------------|
|YourTTS |7.7|
|Vall-E|5.9|
|Moshi|4.7|
|**Voila (w/o LibriSpeech train split)** |**3.2**|
|**Voila (with LibriSpeech train split)** |**2.8**|

_(lower is better)_

# 📝 Citation
If you find our work helpful, please cite us.

```
@article{voila2025,
  author    = {Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, Zhiting Hu},
  title     = {Voila: Voice-Language Foundation Models for Real-Time Autonomous Interaction and Voice Roleplay},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year      = {2025}
}
```
