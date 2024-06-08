import codecs

from setuptools import setup, find_packages

# Define project metadata
NAME = 'audio-denoiser'
VERSION = '0.1.2'
DESCRIPTION = 'A Python library for (speech) audio denoising.'
AUTHOR = 'Jose Solorzano'
URL = 'https://github.com/jose-solorzano/audio-denoiser'

REQUIRES = [
    'numpy>=1.23',
    'tqdm>=4.0.0',
    'torch>=2.1.0',
    'torchaudio>=2.1.0',
    'transformers>=4.28.0',
]

# Long description comes from README
with codecs.open('README.md', 'r', 'utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    url=URL,
    packages=['audio_denoiser', 'audio_denoiser.modules', 'audio_denoiser.helpers'],
    install_requires=REQUIRES,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
    ],
    python_requires='>=3.9',
)
