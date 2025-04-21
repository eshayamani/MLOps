FROM python:3.9

WORKDIR /src

RUN pip install mlflow

EXPOSE 5001

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]