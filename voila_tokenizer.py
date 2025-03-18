import torch
import torchaudio
from torchaudio.functional import resample

from transformers import AutoProcessor, EncodecModel


ALL_BANDWIDTHS = [1.1]

class VoilaTokenizer:
    def __init__(
        self,
        model_path="maitrix-org/Voila-Tokenizer",
        bandwidth_id=0,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.bandwidth = ALL_BANDWIDTHS[bandwidth_id]
        self.bandwidth_id = torch.tensor([bandwidth_id], device=device)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = EncodecModel.from_pretrained(model_path).to(device)

        self.sampling_rate = self.processor.sampling_rate
        self.model_version = self.model.config.model_version


    @torch.no_grad()
    def encode(self, wav, sr):
        wav = torch.tensor(wav, dtype=torch.float32, device=self.device)
        if sr != self.processor.sampling_rate:
            wav = resample(wav, sr, self.processor.sampling_rate)
            sr = self.processor.sampling_rate
        if len(wav.shape) == 1:
            wav = wav[None, None, :]
        elif len(wav.shape) == 2:
            assert wav.shape[0] == 1
            wav = wav[None, :]
        elif len(wav.shape) == 3:
            assert wav.shape[0] == 1 and wav.shape[1] == 1

        # inputs = self.processor(raw_audio=wav, sampling_rate=sr, return_tensors="pt")
        encoder_outputs = self.model.encode(wav, bandwidth=self.bandwidth)
        return encoder_outputs.audio_codes[0, 0]

    @torch.no_grad()
    def decode(self, audio_codes):
        assert len(audio_codes.shape) == 2
        audio_values = self.model.decode(audio_codes[None, None, :, :], [None])[0]
        return audio_values[0, 0]

if __name__ == '__main__':
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str)
    args = parser.parse_args()

    wav, sr = torchaudio.load(args.wav)
    if len(wav.shape) > 1:
        wav = wav[0]

    model = VoilaTokenizer(device="cuda")

    audio_codes = model.encode(wav, sr)
    audio_values = model.decode(audio_codes).cpu().numpy()

    tps = audio_codes.shape[-1] / (audio_values.shape[-1] / model.processor.sampling_rate)
    print(audio_codes.shape, audio_values.shape, tps)
    sf.write("audio_mt.wav", audio_values, model.processor.sampling_rate)
