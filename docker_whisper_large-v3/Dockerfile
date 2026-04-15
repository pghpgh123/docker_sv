FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app
COPY download_model.py /app/download_model.py
COPY start.sh /app/start.sh

EXPOSE 19100

CMD ["sh", "/app/start.sh"]
