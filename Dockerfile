# CPU-only environment for the tests and the full pipeline.
#   docker build -t push-recovery-rl .
#   docker run --rm push-recovery-rl                                         # unit tests
#   docker run --rm -v $PWD/results:/src/results push-recovery-rl ./scripts/run_all.sh
FROM python:3.12-slim
WORKDIR /src
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3", "-m", "pytest", "-q", "tests"]
