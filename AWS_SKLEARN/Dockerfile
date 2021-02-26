FROM aws_sklearn0.23

RUN pip install sagemaker-containers

# Copies the serve code inside the container
COPY serve.py transactions_model.py /opt/ml/code/

# Defines serve.py as script entry point
ENV SAGEMAKER_PROGRAM serve.py