import logging
import os
from typing import List

import click
import numpy as np
import torch
import torchaudio

from config import REPO_ID
from helpers.data_helper import get_file_paths
from helpers.hf_helper import get_hf_token
from helpers.tool_helper import new_exp_id
from training.AudioDenoiserTrainer import AudioDenoiserTrainer

_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
_seed = 1001
_lr = 3e-4
_is_local_run = os.environ.get('KC_LOCAL_RUN', 'False').lower() in ['yes', 'true', '1']
if _is_local_run:
    _batch_size = 4
else:
    _batch_size = 200


def norm_ambient_noise() -> List[torch.Tensor]:
    result = []
    ambient_noise_paths = get_file_paths('/kaggle/input/ambient-noise', '*.wav')
    for an_path in ambient_noise_paths:
        waveform, sample_rate = torchaudio.load(an_path)
        waveform = torch.sum(waveform, dim=0, keepdim=True)
        sd = torch.std(waveform).item()
        waveform = waveform / (sd + 1e-10)
        result.append(waveform)
    return result


def main_impl(num_epochs: int, save_model: bool, hf_token: str = None):
    print(f'Device: {_device}')
    exp_id = new_exp_id()
    rnd = np.random.RandomState(_seed)
    dataset_path_1 = '/kaggle/input/120h-spanish-speech'
    dataset_path_2 = '/kaggle/input/arabic-natural-audio-dataset'
    dataset_path_3 = '/kaggle/input/speaker-recognition-audio-dataset'
    print('Finding audio file paths...')
    audio_paths_1 = get_file_paths(dataset_path_1, '*.wav')
    audio_paths_2 = get_file_paths(dataset_path_2, '*.wav')
    audio_paths_3 = get_file_paths(dataset_path_3, '*.wav')
    audio_paths = audio_paths_1 + audio_paths_2 + audio_paths_3
    rnd.shuffle(audio_paths)
    print(f'Found {len(audio_paths)} audio files.')
    if _is_local_run:
        audio_paths = audio_paths[:10]
        logging.warning(f'Local run: Truncated number of audio paths to {len(audio_paths)}')
    num_valid = min(5000, round(len(audio_paths) * 0.8))
    num_train = len(audio_paths) - num_valid
    train_audio_paths = audio_paths[:num_train]
    valid_audio_paths = audio_paths[num_train:]
    print('Reading ambient noise files...')
    norm_an = norm_ambient_noise()
    trainer = AudioDenoiserTrainer(rnd, train_audio_paths, valid_audio_paths, norm_an,
                                   _device, _batch_size, _lr)
    print('Training...')
    model = trainer.train(num_epochs)
    print('Evaluating...')
    trainer.evaluate(model, num_epochs=3)
    if save_model:
        model.eval()
        model = model.cpu()
        config = dict(model.config)
        config['exp_id'] = exp_id
        if hf_token is None:
            hf_token = get_hf_token()
        model.push_to_hub(REPO_ID, config=config, commit_message=f'Saved model trained in exp-{exp_id}', token=hf_token)
        print(f'Experiment {exp_id} saved a model to {REPO_ID}')
    else:
        print(f'Experiment {exp_id} did not save a model.')


@click.command()
@click.option('-ne', '--num-epochs', required=True, type=int,
              help='Number of epochs')
@click.option('-s', '--save-model', is_flag=True, type=bool, default=False)
def main(num_epochs: int, save_model: bool):
    main_impl(num_epochs=num_epochs, save_model=save_model)


if __name__ == '__main__':
    main()
