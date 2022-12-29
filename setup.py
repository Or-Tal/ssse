from pathlib import Path

from setuptools import setup, find_packages


NAME = 'ssse'
DESCRIPTION = 'Self supervised speech enhancement'

URL = 'https://github.com/fairinternal/audiocraft'
AUTHOR = 'FAIR Speech & Audio'
REQUIRES_PYTHON = '>=3.8.0'

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

REQUIRED = [i.strip() for i in open(HERE / 'requirements.txt') if not i.startswith('#')]

setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require={
        'dev': ['coverage', 'flake8', 'mypy', 'pdoc3', 'pytest', 'types-tqdm'],
    },
    packages=find_packages(),
    package_data={'ssse': ['py.typed']},
    include_package_data=True,
    license='MIT License',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
