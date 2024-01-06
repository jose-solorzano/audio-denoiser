This Python library reduces substantial background noise 
in audio files containing speech. It uses a machine 
learning model (38m parameters) trained to handle 
different types of ambient noise.

## Installation

    pip install audio-denoiser

In Windows, you need the `soundfile` audio backend:

    pip install soundfile

In Linux, both the `soundfile` and `sox` audio backends should be supported. Note that the library is trained with the `soundfile` backend.

## Usage

Basic:

    from audio_denoiser.AudioDenoiser import AudioDenoiser
    
    denoiser = AudioDenoiser()
    in_audio_file = '/content/input-audio-with-noise.wav'
    out_audio_file = '/content/output-denoised-audio.wav'
    denoiser.process_audio_file(in_audio_file, out_audio_file)

With additional options:

    from audio_denoiser.AudioDenoiser import AudioDenoiser
    import torch
    import torchaudio
    
    # Use a CUDA device for inference if available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    denoiser = AudioDenoiser(device=device)
    in_audio_file = '/content/input-audio-with-noise.wav'
    out_audio_file = '/content/output-denoised-audio.wav'
    auto_scale = True # Recommended for low-volume input audio
    denoiser.process_audio_file(in_audio_file, out_audio_file, auto_scale=auto_scale)

You can also provide your own waveform tensor:

    from audio_denoiser.AudioDenoiser import AudioDenoiser

    noisy_waveform, sample_rate = torchaudio.load('/content/input-audio-with-noise.wav')
    denoiser = AudioDenoiser()
    denoised_waveform = denoiser.process_waveform(noisy_waveform, sample_rate, auto_scale=False)
    print('Tensor shape: ', denoised_waveform.shape)
