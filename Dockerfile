FROM python:3.12-slim

# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV PATH="${PATH}:/app/bin"

WORKDIR /app

COPY . .

RUN uv sync

EXPOSE 9000

ENTRYPOINT ["uv","run","webservice.py"]
