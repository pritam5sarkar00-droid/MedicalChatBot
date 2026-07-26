FROM python:3.10-slim-bookworm

# This is the all-in-one image: the main app AND the embedding/reranker
# models in a single process/container -- the simplest possible
# deployment (one service, one free-tier instance). It's also what's
# used for local `docker build`/`docker run`.
#
# For the split deployment instead (a separate, much lighter API
# container plus a dedicated inference_service/ container -- see
# DEPLOYMENT.md for why and how), build Dockerfile.api at the repo root
# for the API side, and inference_service/Dockerfile for the model side.

WORKDIR /app

COPY . /app

# torch (a dependency of sentence-transformers, used for local embeddings)
# defaults to a CUDA-enabled build on Linux even though this app never
# touches a GPU -- that pulls in ~1GB+ of unused NVIDIA/CUDA packages
# (cublas, cudnn, cufft, triton, ...) on top of torch's own ~800MB, which
# is most of what "pip install" feeling slow, flaky, or running out of
# disk on this project usually turns out to be. Installing the CPU-only
# build first means the later requirements install finds torch already
# satisfied and never reaches for the CUDA build at all.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-local-inference.txt

# Default port for local/AWS use. Free hosts (Render, Koyeb) inject their
# own PORT env var at runtime, which app.py reads automatically.
ENV PORT=8080
EXPOSE 8080

CMD ["python3", "app.py"]
