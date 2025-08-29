# CPU base to avoid CUDA hassles
FROM pytorch/pytorch:2.2.2-cpu
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .[dev]
CMD ["bash"]
