FROM python:3.11-slim
RUN pip install --no-cache-dir pandas==2.3.3 numpy==1.26.4 matplotlib==3.7.5
WORKDIR /app
CMD ["sleep", "infinity"]
