from setuptools import setup, find_packages

setup(
    name='transformer-training-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for training transformer models.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'pyyaml',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'train=script.train:main',
        ],
    },
)