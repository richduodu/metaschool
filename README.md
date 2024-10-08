# AI Tutor with Active Learning

## Overview

This AI Tutor project is designed to assist students by explaining educational concepts and helping them solve problems interactively. It leverages NVIDIA Workbench, PyTorch, and Hugging Face Transformers to achieve these functionalities. The project also interfaces with NVIDIA's Gemini SDK for optimized performance on NVIDIA GPUs.

## Features

1. **Question Explanation**: Ask any educational question, and the tutor will provide an explanation.
2. **Interactive Problem Solving**: Upload a question, and the tutor will work with the student to find a solution step-by-step.
3. **Guardrails**: Ensures that responses are educational and encourages active learning.
4. **Gemini Integration**: Optimized for inference on NVIDIA GPUs using the Gemini SDK.
5. **Web Interface**: A simple web-based UI to ask questions and solve problems.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/richduodu/metaschool.git
   cd metaschool

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA installed
- PyTorch 2.0+
- Hugging Face Transformers
- Flask or FastAPI
- Gemini SDK

### Setup

1. Clone the repository
2. ```bash
    cd metaschool
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
