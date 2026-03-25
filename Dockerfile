FROM python:3.10-slim

# Define the Build Argument
ARG RUN_ID

# Set environment variables
ENV MODEL_DIR=/opt/ml/model

# Create model directory
RUN mkdir -p ${MODEL_DIR}

# Mocking the download for the assignment
RUN echo "Simulating download for Run ID: ${RUN_ID}" > ${MODEL_DIR}/model_status.txt

CMD ["echo", "Model deployment simulated successfully"]