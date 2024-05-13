import functools
import os
import platform
import re
import subprocess
import sys
from enum import Enum
from typing import List, Tuple
from urllib.parse import quote, urlparse

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


class SetupSpec:
    def __init__(self) -> None:
        self.extras: dict = {}
        self.install_requires: List[str] = []

    @property
    def unique_extras(self) -> dict[str, list[str]]:
        unique_extras = {}
        for k, v in self.extras.items():
            unique_extras[k] = list(set(v))
        return unique_extras


class AVXType(Enum):
    BASIC = "basic"
    AVX = "AVX"
    AVX2 = "AVX2"
    AVX512 = "AVX512"

    @staticmethod
    def of_type(avx: str):
        for item in AVXType:
            if item._value_ == avx:
                return item
        return None


class OSType(Enum):
    WINDOWS = "win"
    LINUX = "linux"
    DARWIN = "darwin"
    OTHER = "other"


@functools.cache
def get_cpu_avx_support() -> Tuple[OSType, AVXType]:
    system = platform.system()
    os_type = OSType.OTHER
    cpu_avx = AVXType.BASIC
    env_cpu_avx = AVXType.of_type(os.getenv("DBGPT_LLAMA_CPP_AVX"))

    if "windows" in system.lower():
        os_type = OSType.WINDOWS
        output = "avx2"
        print("Current platform is windows, use avx2 as default cpu architecture")
    elif system == "Linux":
        os_type = OSType.LINUX
        result = subprocess.run(
            ["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout.decode()
    elif system == "Darwin":
        os_type = OSType.DARWIN
        result = subprocess.run(
            ["sysctl", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout.decode()
    else:
        os_type = OSType.OTHER
        print("Unsupported OS to get cpu avx, use default")
        return os_type, env_cpu_avx if env_cpu_avx else cpu_avx

    if "avx512" in output.lower():
        cpu_avx = AVXType.AVX512
    elif "avx2" in output.lower():
        cpu_avx = AVXType.AVX2
    elif "avx " in output.lower():
        # cpu_avx =  AVXType.AVX
        pass
    return os_type, env_cpu_avx if env_cpu_avx else cpu_avx



def get_cuda_version_from_torch():
    try:
        import torch

        return torch.version.cuda
    except:
        return None


def get_cuda_version_from_nvcc():
    try:
        output = subprocess.check_output(["nvcc", "--version"])
        version_line = [
            line for line in output.decode("utf-8").split("\n") if "release" in line
        ][0]
        return version_line.split("release")[-1].strip().split(",")[0]
    except:
        return None


def get_cuda_version_from_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", output)
        if match:
            return match.group(1)
        else:
            return None
    except:
        return None


def get_cuda_version() -> str:
    try:
        cuda_version = get_cuda_version_from_torch()
        if not cuda_version:
            cuda_version = get_cuda_version_from_nvcc()
        if not cuda_version:
            cuda_version = get_cuda_version_from_nvidia_smi()
        return cuda_version
    except Exception:
        return None


def encode_url(package_url: str) -> str:
    parsed_url = urlparse(package_url)
    encoded_path = quote(parsed_url.path)
    safe_url = parsed_url._replace(path=encoded_path).geturl()
    return safe_url, parsed_url.path


setup_spec = SetupSpec()


def llama_cpp_python_cuda_requires():
    cuda_version = get_cuda_version()
 
    device = "cpu"
    if not cuda_version:
        print("CUDA not support, use cpu version")
        return

    os_type, cpu_avx = get_cpu_avx_support()
    print(f"OS: {os_type}, cpu avx: {cpu_avx}")
    supported_os = [OSType.WINDOWS, OSType.LINUX]
    if os_type not in supported_os:
        print(
            f"llama_cpp_python_cuda just support in os: {[r._value_ for r in supported_os]}"
        )
        return

    base_url = "https://github.com/abetlen/llama-cpp-python/releases/download"
    llama_cpp_version = "0.2.73"
    py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    os_pkg_name = "linux_x86_64" if os_type == OSType.LINUX else "win_amd64"
    extra_index_url = f"{base_url}/v{llama_cpp_version}-cu124/llama_cpp_python-{llama_cpp_version}-{py_version}-{py_version}-{os_pkg_name}.whl"
   
    extra_index_url, _ = encode_url(extra_index_url)
    print(f"Install llama_cpp_python_cuda from {extra_index_url}")
    
    requirements.remove('llama-cpp-python')
    requirements.append(f"llama_cpp_python @ {extra_index_url}")
   
llama_cpp_python_cuda_requires()

setup(
    name="prompt_generator",
    version="0.0.1",
    description="A small package to create prompt",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/WaveGenAI/prompt-generator.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,

)
