FROM python:3.13-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/adampetrovic/abb-cvc-extract"
LABEL org.opencontainers.image.description="Extract time-series data from Aussie Broadband CVC capacity graph images"

COPY --from=builder /install /usr/local
COPY abb_cvc_extract.py /app/abb_cvc_extract.py

ENTRYPOINT ["python3", "/app/abb_cvc_extract.py"]
