FROM python:3.10-slim

ARG RUN_ID

ENV MODEL_DIR=/opt/ml/model
RUN mkdir -p ${MODEL_DIR}
RUN echo "Simulating download for Run ID: ${RUN_ID}" > ${MODEL_DIR}/model_status.txt

CMD ["echo", "Model deployment simulated successfully"]