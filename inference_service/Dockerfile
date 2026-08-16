FROM python:3.10-slim-bookworm

WORKDIR /app

COPY inference_service/requirements.txt /app/requirements.txt

# CPU-only torch first -- see the root Dockerfile's comment for why this
# matters (skips ~1-2GB of unused CUDA/NVIDIA packages that a plain
# `pip install sentence-transformers` would otherwise pull in on Linux).
RUN pip install --no-cache-dir "typing-extensions>=4.10.0"
RUN pip install --no-cache-dir --timeout 100 --retries 10 torch --index-url  https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_service/app.py /app/app.py

# Both models download from HuggingFace's hub on first startup and are
# cached under /root/.cache/huggingface/ for every run after -- on a
# free-tier host with an ephemeral filesystem, that cache doesn't survive
# a redeploy, so expect the ~10-40s model-load cost (see app.py's
# _load_models()) on every redeploy, not just the very first one. It does
# survive a plain restart/cold-start-from-sleep on hosts like Render/
# Koyeb, since those keep the same container filesystem.
ENV PORT=8081
EXPOSE 8081

CMD ["python3", "app.py"]
