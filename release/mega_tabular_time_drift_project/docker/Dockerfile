# Minimal Dockerfile
FROM mambaorg/micromamba:1.5.8
WORKDIR /app

COPY environment.yml /app/environment.yml
RUN micromamba env create -f /app/environment.yml -y &&     micromamba clean --all --yes
SHELL ["/bin/bash", "-lc"]
RUN echo "source activate mega-tabular" >> ~/.bashrc

COPY . /app
EXPOSE 8000
CMD ["bash", "-lc", "source activate mega-tabular && uvicorn src.serve:app --host 0.0.0.0 --port 8000"]
