from setuptools import find_packages, setup

setup(
    name="debategpt",
    description="Train and use DebateGPT, a language model designed to simulate debates.",
    author="paulbricman",
    packages=["debategpt", "debategpt.training", "debategpt.inference"],
)
