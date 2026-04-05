FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download CIFAR-10 at build time (optional – can also happen at runtime)
RUN python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', download=True)"

CMD ["bash"]
