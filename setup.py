from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="prompt_generator",
    version="0.0.1",
    description="A small package to create prompt",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/WaveGenAI/lyrics.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True
)
