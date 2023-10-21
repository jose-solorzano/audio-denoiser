import torch
import torchaudio
import transformers

from audio_denoiser.AudioDenoiser import AudioDenoiser

if __name__ == '__main__':
    denoiser = AudioDenoiser()
    print('Model config: ', denoiser.model.config)
    print('Torch: ', torch.__version__)
    print('Torchaudio: ', torchaudio.__version__)
    print('Transformers: ', transformers.__version__)
    in_audio_file = '/data/audio-example-1-out.wav'
    out_audio_file = '/data/audio-example-1-out-denoised-v2.wav'
    denoiser.process_audio_file(in_audio_file, out_audio_file)
    print(f'Wrote {out_audio_file}')
