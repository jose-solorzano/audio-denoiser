from audio_denoiser.modules.AudioNoiseModel import AudioNoiseModel

if __name__ == '__main__':
    model = AudioNoiseModel.from_pretrained('jose-h-solorzano/audio-denoiser-512-32-v1')
    print(model.scaler)
