import os
from audio_denoiser.AudioDenoiser import AudioDenoiser

if __name__ == '__main__':
    dir_name = '/data/audio'
    denoiser = AudioDenoiser()
    for i in range(2):
        file_path = os.path.join(dir_name, f'noisy-example-{i+1}.wav')
        out_path = os.path.join(dir_name, f'denoised-example-{i+1}.wav')
        denoiser.process_audio_file(file_path, out_path, auto_scale=True)
        print(f'Wrote {out_path}')
