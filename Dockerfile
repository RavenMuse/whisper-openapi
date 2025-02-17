FROM python:3.12-slim

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="${PATH}:/app/bin"

WORKDIR /app

COPY . .

RUN uv sync --upgrade

EXPOSE 9000

ENTRYPOINT ["uv"，"run","webservice.py"]
