from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="debategpt",
    version="0.0.2",
    description="Train and use DebateGPT, a language model designed to simulate debates.",
    author="paulbricman",
    author_email="paulbricman@protonmail.com",
    packages=["debategpt", "debategpt.training", "debategpt.inference"],
    install_requires=[
        "transformers",
        "networkx",
        "torch",
        "sentencepiece",
        "protobuf==3.20.0",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cpu",
    ]
)
