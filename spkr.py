import torch
import torchaudio
from torchaudio.functional import resample

from pyannote.audio import Model
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


class SpeakerEmbedding:
    def __init__(self, model_path="pyannote/wespeaker-voxceleb-resnet34-LM", device="cuda"):
        model = Model.from_pretrained(model_path).eval()

        self.device = torch.device(device)
        self.sample_rate = 16000
        self.model = model.to(self.device)

    @torch.no_grad()
    def __call__(self, wav, sr):
        wav = torch.tensor(wav, device=self.device)
        if sr != self.sample_rate:
            wav = resample(wav, sr, self.sample_rate)
            sr = self.sample_rate

        assert len(wav.shape) <= 3
        is_batch = False
        if len(wav.shape) == 3:
            is_batch = True
        elif len(wav.shape) == 2:
            wav = wav[None, :, :]
        else:
            wav = wav[None, None, :]

        with torch.inference_mode():
            embeddings = self.model(wav)

        if is_batch:
            return embeddings
        else:
            return embeddings[0]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    model = SpeakerEmbedding(device="cuda")

    wav, sr = torchaudio.load(args.wav)
    print(model(wav, sr))
