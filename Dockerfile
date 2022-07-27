FROM python:3.9-rc-buster

WORKDIR /code

ENV POETRY_VERSION=1.1.13

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install

COPY word2vect train.py data ./

CMD ["python", "train.py"]
