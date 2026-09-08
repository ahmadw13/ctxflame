FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY ctxflame/ ./ctxflame/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["ctxflame"]
CMD ["--help"]
