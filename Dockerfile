FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -L https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:stable
ENV PATH="/root/.elan/bin:${PATH}"
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN lake new aidan_lean_project math || true
WORKDIR /app/aidan_lean_project

RUN lake exe cache get || true
RUN lake update
RUN lake exe cache get
#echo kısmı silindi.

# ram için modüler derleme
RUN lake build Mathlib.Tactic Mathlib.Algebra.Group.Basic

WORKDIR /app
RUN mkdir -p /app/model_data

ENV LEAN_PROJECT_PATH="/app/aidan_lean_project"
CMD ["python3", "main.py"]