import io
import copy
import librosa
import numpy as np


AUDIO_TOKEN_FORMAT = "<|{}|>"

DEFAULT_SYSTEM_START_TOKEN = "<SYSTEM>"
DEFAULT_SYSTEM_END_TOKEN   = "</SYSTEM>"

DEFAULT_TTS_REF_START_TOKEN = "<au_tts_ref_start>"
DEFAULT_TTS_REF_END_TOKEN   = "<au_tts_ref_end>"
DEFAULT_TTS_REF_TOKEN = "<au_tts_ref>"

DEFAULT_CHAT_REF_START_TOKEN = "<au_chat_ref_start>"
DEFAULT_CHAT_REF_END_TOKEN   = "<au_chat_ref_end>"
DEFAULT_CHAT_REF_TOKEN = "<au_chat_ref>"

DEFAULT_HUMAN_TOKEN = "<|HUMAN|>"
DEFAULT_ASSISTANT_TOKEN = "<|VOILA|>"

DEFAULT_AUDIO_TOKEN = "<au_token>"

# ===================================
# task special token
# -----------------------------------
TASK_ASR_TOKEN = "<asr>"
TASK_TTS_TOKEN = "<tts>"
TASK_CHAT_TOKEN = "<chat>"
TASK_STREAM_CHAT_TOKEN = "<stream_chat>"

TASK_ASR_TEXT_OUTPUT = "<asr_text_output>"
TASK_TTS_AUDIO_OUTPUT = "<tts_audio_output>"
TASK_CHAT_TEXT_OUTPUT = "<chat_text_output>"
TASK_CHAT_AUDIO_OUTPUT = "<chat_audio_output>"

CHAT_AUDIO_TEXT_SPLIT_TOKEN = "<chat_audio_text_split>"
# ===================================

PREPEND_LEN = 80
SEG_LEN = 640
AUDIO_SR = 16000

TASK_TYPE_CONF = {
    "chat_asr": TASK_ASR_TOKEN + TASK_ASR_TEXT_OUTPUT,
    "chat_tts": TASK_TTS_TOKEN + TASK_TTS_AUDIO_OUTPUT,
    "chat_tito": TASK_CHAT_TOKEN + TASK_CHAT_TEXT_OUTPUT,
    "chat_tiao": TASK_CHAT_TOKEN + TASK_CHAT_AUDIO_OUTPUT,
    "chat_aiao": TASK_CHAT_TOKEN + TASK_CHAT_AUDIO_OUTPUT,
    "chat_atiao": TASK_CHAT_TOKEN + TASK_CHAT_AUDIO_OUTPUT,
    "chat_aiao_auto": TASK_STREAM_CHAT_TOKEN + TASK_CHAT_AUDIO_OUTPUT,
}


def _get_zero_audio_pad(token_num):
    return np.zeros(SEG_LEN*token_num)

def _wrapper_audio_tokens(audio_tokens, num_codebooks, codebook_size):
    ret_audio_tokens = []
    for n in range(num_codebooks):
        audio_token = audio_tokens[n]
        ret_audio_tokens.append(''.join([AUDIO_TOKEN_FORMAT.format(au + n*codebook_size) if isinstance(au, int) else au for au in audio_token]))
    return ret_audio_tokens

def _wrapper_audio_tokens_autonomous(audio_tokens, num_codebooks, codebook_size, audio_token_min_id):
    ret_audio_tokens = []
    for n in range(num_codebooks):
        audio_token = audio_tokens[n]
        ret_audio_tokens.append([(au + n*codebook_size + audio_token_min_id) for au in audio_token])
    return ret_audio_tokens

# Item format
# {
#   "instruction": "",
#   "conversations": [
#     {
#       "from": "user" or "assistant",
#       "text": "",
#       "audio": {
#         "array": [],
#         "sr": 16000,
#         "bytes": "",
#         "file": "",
#       },
#     }
#   ],
# }
def _token_input_format(item, tokenizer, tokenizer_voila, dataset_cfg):
    task_type = dataset_cfg["task_type"]
    num_codebooks = dataset_cfg["num_codebooks"]
    codebook_size = dataset_cfg["codebook_size"]

    task_token = TASK_TYPE_CONF[task_type]

    # Construct system message
    system = item["instruction"]
    if task_type in ["chat_aiao", "chat_atiao", "chat_tiao"]:
        system = DEFAULT_CHAT_REF_START_TOKEN + DEFAULT_CHAT_REF_TOKEN + DEFAULT_CHAT_REF_END_TOKEN + system
    elif task_type == "chat_tts":
        system = DEFAULT_TTS_REF_START_TOKEN + DEFAULT_TTS_REF_TOKEN + DEFAULT_TTS_REF_END_TOKEN + system
    else:
        print (f"task type {task_type} do not use ref.")
    system = task_token + system
    system = DEFAULT_SYSTEM_START_TOKEN + system + DEFAULT_SYSTEM_END_TOKEN

    # Get ids for system
    system_ids = tokenizer.encode(system, add_special_tokens=False)

    # Copy into num_codebooks input ids
    input_ids_list = []
    for _ in range(num_codebooks):
        input_ids_list.append(copy.deepcopy(system_ids))

    # Assemble conversations
    for i, turn in enumerate(item["conversations"]):
        if turn['from'] == 'assistant':
            # task with audio token as input, prepare audio token
            if task_type in ["chat_aiao", "chat_tts"]:
                if "audio" not in turn:
                    content = DEFAULT_ASSISTANT_TOKEN
                    content_ids = tokenizer.encode(content, add_special_tokens=False)
                    for n in range(num_codebooks):
                        input_ids_list[n] += copy.deepcopy(content_ids)
                else:
                    # Load audio
                    if 'array' in turn['audio']:
                        assert "sr" in turn["audio"]
                        if len(turn["audio"]['array'].shape) > 1:
                            assert turn["audio"]['array'].shape[0] <= 2
                            turn["audio"]['array'] = librosa.to_mono(turn["audio"]['array'])
                        audio = librosa.resample(turn["audio"]['array'], orig_sr=turn["audio"]["sr"], target_sr=AUDIO_SR)
                    elif "bytes" in turn['audio']:
                        audio, _ = librosa.load(io.BytesIO(turn["audio"]['bytes']), sr=AUDIO_SR)
                    elif "file" in turn['audio']:
                        audio, _ = librosa.load(turn["audio"]['file'], sr=AUDIO_SR)
                    else:
                        raise Exception(f"No audio input for task {task_type}")

                    # get audio token
                    audio_tokens = tokenizer_voila.encode(audio, sr=AUDIO_SR)
                    audio_tokens = audio_tokens.cpu().numpy().tolist()
                    audio_tokens = _wrapper_audio_tokens(audio_tokens, num_codebooks, codebook_size)

                    for n in range(num_codebooks):
                        content = DEFAULT_ASSISTANT_TOKEN + audio_tokens[n] + tokenizer.eos_token
                        content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                                max_length=tokenizer.model_max_length)
                        input_ids_list[n] += content_ids

            elif task_type in ["chat_tito", "chat_asr"]:
                if "text" not in turn:
                    content = DEFAULT_ASSISTANT_TOKEN
                    content_ids = tokenizer.encode(content, add_special_tokens=False)
                    for n in range(num_codebooks):
                        input_ids_list[n] += copy.deepcopy(content_ids)
                else:
                    text = turn['text'].strip()
                    content = DEFAULT_ASSISTANT_TOKEN + text + tokenizer.eos_token
                    content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                                    max_length=tokenizer.model_max_length)
                    for n in range(num_codebooks):
                        input_ids_list[n] += copy.deepcopy(content_ids)
            else:
                raise ValueError (f"[Error] Invalid data type of {task_type}.")
        else:
            # task with audio token as input, prepare audio token
            if task_type in ["chat_aiao", "chat_asr"]:
                # Load audio
                assert "audio" in turn
                if 'array' in turn['audio']:
                    assert "sr" in turn["audio"]
                    if len(turn["audio"]['array'].shape) > 1:
                        assert turn["audio"]['array'].shape[0] <= 2
                        turn["audio"]['array'] = librosa.to_mono(turn["audio"]['array'])
                    audio = librosa.resample(turn["audio"]['array'], orig_sr=turn["audio"]["sr"], target_sr=AUDIO_SR)
                elif "bytes" in turn['audio']:
                    audio, _ = librosa.load(io.BytesIO(turn["audio"]['bytes']), sr=AUDIO_SR)
                elif "file" in turn['audio']:
                    audio, _ = librosa.load(turn["audio"]['file'], sr=AUDIO_SR)
                else:
                    raise Exception(f"No audio input for task {task_type}")

                # get audio token
                audio_tokens = tokenizer_voila.encode(audio, sr=AUDIO_SR)
                audio_tokens = audio_tokens.cpu().numpy().tolist()
                audio_tokens = _wrapper_audio_tokens(audio_tokens, num_codebooks, codebook_size)

                for n in range(num_codebooks):
                    content = DEFAULT_HUMAN_TOKEN + audio_tokens[n]
                    content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                            max_length=tokenizer.model_max_length)
                    input_ids_list[n] += copy.deepcopy(content_ids)
            elif task_type in ["chat_tito", "chat_tts"]:
                text = turn['text'].strip()
                content = DEFAULT_HUMAN_TOKEN + text
                content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                                max_length=tokenizer.model_max_length)
                for n in range(num_codebooks):
                    input_ids_list[n] += copy.deepcopy(content_ids)
            else:
                raise ValueError (f"[Error] Invalid data type of {task_type}.")

    for n in range(num_codebooks):
        input_ids_list[n] = input_ids_list[n][:tokenizer.model_max_length]

    return input_ids_list, None, None, None

def _token_input_format_autonomous(item, tokenizer, tokenizer_voila, dataset_cfg):
    task_type = dataset_cfg["task_type"]
    num_codebooks = dataset_cfg["num_codebooks"]
    codebook_size = dataset_cfg["codebook_size"]
    assert task_type == "chat_aiao_auto", f"only support chat_aiao_auto, {task_type} is invalid"

    DEFAULT_HUMAN_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_HUMAN_TOKEN)
    assert isinstance(DEFAULT_HUMAN_TOKEN_ID, int), "DEFAULT_HUMAN_TOKEN_ID should be an integer"
    AUDIO_MIN_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(0))
    assert isinstance(AUDIO_MIN_TOKEN_ID, int), "AUDIO_MIN_TOKEN_ID should be an integer"

    task_token = TASK_TYPE_CONF[task_type]

    # Construct system message
    system = DEFAULT_CHAT_REF_START_TOKEN + DEFAULT_CHAT_REF_TOKEN + DEFAULT_CHAT_REF_END_TOKEN
    system = task_token + system
    system = DEFAULT_SYSTEM_START_TOKEN + system + DEFAULT_SYSTEM_END_TOKEN

    # Get ids for system
    system_ids_list = [[], []]
    system_ids = tokenizer.encode(system, add_special_tokens=False)

    # Insert instruction tokens into system prompt tokens
    instruction = item["instruction"]
    if instruction != "":
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
    else:
        instruction_ids = []

    system_ids_list[0] = system_ids[:-1] + instruction_ids + system_ids[-1:]
    system_ids_list[1] = system_ids[:-1] + instruction_ids + system_ids[-1:]

    # Copy into num_codebooks input ids
    channel1_input_ids_list = [[] for _ in range(num_codebooks)]
    channel2_input_ids_list = [[] for _ in range(num_codebooks)]
    for n in range(num_codebooks):
        channel1_input_ids_list[n] += copy.deepcopy(system_ids_list[0]) + [DEFAULT_HUMAN_TOKEN_ID]
        channel2_input_ids_list[n] += copy.deepcopy(system_ids_list[1]) + [DEFAULT_HUMAN_TOKEN_ID]

    # prepare audio token to simulate streaming input
    audio_meta = item['conversations'][0]['audio']
    if 'array' in audio_meta:
        assert "sr" in audio_meta
        if len(audio_meta['array'].shape) > 1:
            assert audio_meta['array'].shape[0] <= 2
            audio_meta['array'] = librosa.to_mono(audio_meta['array'])
        audio = librosa.resample(audio_meta['array'], orig_sr=audio_meta["sr"], target_sr=AUDIO_SR)
    elif "bytes" in audio_meta:
        audio, _ = librosa.load(io.BytesIO(audio_meta['bytes']), sr=AUDIO_SR)
    elif "file" in audio_meta:
        audio, _ = librosa.load(audio_meta['file'], sr=AUDIO_SR)
    else:
        raise Exception(f"No audio input for task {task_type}")

    # get audio token
    streaming_user_input_audio_tokens = tokenizer_voila.encode(audio, sr=AUDIO_SR)
    streaming_user_input_audio_tokens = streaming_user_input_audio_tokens.cpu().numpy().tolist()
    streaming_user_input_audio_tokens = _wrapper_audio_tokens_autonomous(streaming_user_input_audio_tokens, num_codebooks, codebook_size, AUDIO_MIN_TOKEN_ID)

    return [channel1_input_ids_list, channel2_input_ids_list], None, None, streaming_user_input_audio_tokens

def _alpha_audio_input_format(item, tokenizer, dataset_cfg):
    task_type = dataset_cfg["task_type"]
    num_codebooks = dataset_cfg["num_codebooks"]
    codebook_size = dataset_cfg["codebook_size"]

    task_token = TASK_TYPE_CONF[task_type]

    # Construct system message
    system = item["instruction"]
    if task_type in ["chat_aiao", "chat_atiao", "chat_tiao"]:
        system = DEFAULT_CHAT_REF_START_TOKEN + DEFAULT_CHAT_REF_TOKEN + DEFAULT_CHAT_REF_END_TOKEN + system
    elif task_type == "chat_tts":
        system = DEFAULT_TTS_REF_START_TOKEN + DEFAULT_TTS_REF_TOKEN + DEFAULT_TTS_REF_END_TOKEN + system
    else:
        print (f"task type {task_type} do not use ref.")
    system = task_token + system
    system = DEFAULT_SYSTEM_START_TOKEN + system + DEFAULT_SYSTEM_END_TOKEN

    # Get ids for system
    system_ids = tokenizer.encode(system, add_special_tokens=False)

    # Copy into num_codebooks input ids
    input_ids_list = []
    for _ in range(num_codebooks):
        input_ids_list.append(copy.deepcopy(system_ids))

    # Construct audio data and mask
    audio_data = [np.array([0]*PREPEND_LEN)]
    audio_data.append(_get_zero_audio_pad(len(system_ids)))
    audio_data_mask = [0] * len(system_ids)

    # Assemble conversations
    for i, turn in enumerate(item["conversations"]):
        if turn['from'] == 'assistant':
            # task with audio token as input, prepare audio token
            if task_type in ["chat_aiao"]:
                if "audio" not in turn:
                    content = DEFAULT_ASSISTANT_TOKEN
                    content_ids = tokenizer.encode(content, add_special_tokens=False)
                    for n in range(num_codebooks):
                        input_ids_list[n] += copy.deepcopy(content_ids)
                    # preprocess audio_data & audio_data_mask
                    audio_data.append(_get_zero_audio_pad(len(content_ids)))
                    audio_data_mask += [0] * len(content_ids)
                else:
                    # Load audio
                    if 'array' in turn['audio']:
                        assert "sr" in turn["audio"]
                        if len(turn["audio"]['array'].shape) > 1:
                            assert turn["audio"]['array'].shape[0] <= 2
                            turn["audio"]['array'] = librosa.to_mono(turn["audio"]['array'])
                        audio = librosa.resample(turn["audio"]['array'], orig_sr=turn["audio"]["sr"], target_sr=AUDIO_SR)
                    elif "bytes" in turn['audio']:
                        audio, _ = librosa.load(io.BytesIO(turn["audio"]['bytes']), sr=AUDIO_SR)
                    elif "file" in turn['audio']:
                        audio, _ = librosa.load(turn["audio"]['file'], sr=AUDIO_SR)
                    else:
                        raise Exception(f"No audio input for task {task_type}")

                    # get audio token
                    audio_token_num = int(len(audio) / SEG_LEN)
                    audio_token = [DEFAULT_AUDIO_TOKEN] * audio_token_num
                    audio_token = ''.join(audio_token)
                    audio = audio[:SEG_LEN*audio_token_num]             # trim audio

                    content = DEFAULT_ASSISTANT_TOKEN + audio_token + tokenizer.eos_token
                    content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                                max_length=tokenizer.model_max_length)
                    for n in range(num_codebooks):
                        input_ids_list[n] += copy.deepcopy(content_ids)

                    audio_data.append(_get_zero_audio_pad(1))
                    audio_data_mask += [0]
                    audio_data.append(audio)
                    audio_data_mask += [1] * audio_token_num
                    audio_data.append(_get_zero_audio_pad(1))
                    audio_data_mask += [0]
            elif task_type in ["chat_tito"]:
                if "text" not in turn:
                    content = DEFAULT_ASSISTANT_TOKEN
                    content_ids = tokenizer.encode(content, add_special_tokens=False)
                    for n in range(num_codebooks):
                        input_ids_list[n] += copy.deepcopy(content_ids)
                    # preprocess audio_data & audio_data_mask
                    audio_data.append(_get_zero_audio_pad(len(content_ids)))
                    audio_data_mask += [0] * len(content_ids)
                else:
                    text = turn['text'].strip()
                    content = DEFAULT_ASSISTANT_TOKEN + text + tokenizer.eos_token
                    content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                                    max_length=tokenizer.model_max_length)
                    for n in range(num_codebooks):
                        input_ids_list[n] += copy.deepcopy(content_ids)
                    audio_data.append(_get_zero_audio_pad(len(content_ids)))
                    audio_data_mask += [0] * len(content_ids)
            else:
                raise ValueError (f"[Error] Invalid data type of {task_type}.")
        else:
            # task with audio token as input, prepare audio token
            if task_type in ["chat_aiao"]:
                # Load audio
                assert "audio" in turn
                if 'array' in turn['audio']:
                    assert "sr" in turn["audio"]
                    if len(turn["audio"]['array'].shape) > 1:
                        assert turn["audio"]['array'].shape[0] <= 2
                        turn["audio"]['array'] = librosa.to_mono(turn["audio"]['array'])
                    audio = librosa.resample(turn["audio"]['array'], orig_sr=turn["audio"]["sr"], target_sr=AUDIO_SR)
                elif "bytes" in turn['audio']:
                    audio, _ = librosa.load(io.BytesIO(turn["audio"]['bytes']), sr=AUDIO_SR)
                elif "file" in turn['audio']:
                    audio, _ = librosa.load(turn["audio"]['file'], sr=AUDIO_SR)
                else:
                    raise Exception(f"No audio input for task {task_type}")

                # get audio token
                audio_token_num = int(len(audio) / SEG_LEN)
                audio_token = [DEFAULT_AUDIO_TOKEN] * audio_token_num
                audio_token = ''.join(audio_token)
                audio = audio[:SEG_LEN*audio_token_num]             # trim audio

                content = DEFAULT_HUMAN_TOKEN + audio_token
                content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                            max_length=tokenizer.model_max_length)
                for n in range(num_codebooks):
                    input_ids_list[n] += copy.deepcopy(content_ids)

                audio_data.append(_get_zero_audio_pad(1))
                audio_data_mask += [0]
                audio_data.append(audio)
                audio_data_mask += [1] * audio_token_num
            elif task_type in ["chat_tito"]:
                text = turn['text'].strip()
                content = DEFAULT_HUMAN_TOKEN + text
                content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True,
                                                max_length=tokenizer.model_max_length)
                for n in range(num_codebooks):
                    input_ids_list[n] += copy.deepcopy(content_ids)
                audio_data.append(_get_zero_audio_pad(len(content_ids)))
                audio_data_mask += [0] * len(content_ids)
            else:
                raise ValueError (f"[Error] Invalid data type of {task_type}.")

    for n in range(num_codebooks):
        input_ids_list[n] = input_ids_list[n][:tokenizer.model_max_length]
    audio_data_mask = audio_data_mask[:tokenizer.model_max_length]
    audio_data = np.concatenate(audio_data)
    audio_data = audio_data[:PREPEND_LEN + tokenizer.model_max_length*SEG_LEN]

    return input_ids_list, audio_data, audio_data_mask, None

# Item format
# {
#   "instruction": "",
#   "conversations": [
#     {
#       "from": "user" or "assistant",
#       "text": "",
#       "audio": {
#         "array": [],
#         "sr": 16000,
#         "bytes": "",
#         "file": "",
#       },
#     }
#   ],
# }
def voila_input_format(item, tokenizer, tokenizer_voila, dataset_cfg):
    if dataset_cfg["input_type"] == "audio":
        return _alpha_audio_input_format(item, tokenizer, dataset_cfg)
    elif dataset_cfg["input_type"] == "autonomous":
        return _token_input_format_autonomous(item, tokenizer, tokenizer_voila, dataset_cfg)
    else:
        return _token_input_format(item, tokenizer, tokenizer_voila, dataset_cfg)
