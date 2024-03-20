FROM python:3.8-slim as base

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 3001

CMD ["python", "main.py"]