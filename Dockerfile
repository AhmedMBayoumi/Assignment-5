FROM python:3.10-slim

ARG RUN_ID
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD

ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME
ENV MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install mlflow

ENV MODEL_DIR=/opt/ml/model
RUN mkdir -p ${MODEL_DIR}

RUN mlflow artifacts download --run-id ${RUN_ID} --artifact-path "" --dst-path ${MODEL_DIR}

CMD ["echo", "Model deployment successful. Model downloaded for Run ID: ${RUN_ID}"]