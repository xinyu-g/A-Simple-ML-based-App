FROM python:3.10.8


# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/ml_api

# ENV PYTHONPATH=/WORKDIR

ARG PIP_EXTRA_INDEX_URL

# Install requirements, including from Gemfury
ADD ./* /opt/ml_api


# RUN user=echo $(python3 -m site --user-base)
# RUN python -m ensurepip --upgrade
RUN pip install --upgrade pip
# RUN which pip
# COPY requirements.txt /
# RUN pip3 freeze > /Users/xinyu/repos/Fetch/ML/packages/api/requirements.txt
# RUN pip install --ignore-installed -r /opt/ml_api/requirements.txt
RUN pip install --ignore-installed uvicorn==0.18.2
RUN pip install --ignore-installed fastapi==0.79.0
RUN pip install --ignore-installed python-multipart==0.0.5
RUN pip install --ignore-installed pydantic==1.9.1
RUN pip install --ignore-installed typing_extensions==3.10.0
RUN pip install --ignore-installed loguru==0.6.0
RUN pip install --ignore-installed pandas==1.4.3
RUN pip install --ignore-installed numpy==1.23.2
RUN pip install --ignore-installed torch==1.13.0
RUN pip install --ignore-installed matplotlib==3.5.3
RUN pip install --ignore-installed seaborn==0.12.1
RUN pip install --ignore-installed scikit-learn==1.1.3



RUN chmod +x /opt/ml_api/run.sh
RUN chown -R ml-api-user:ml-api-user ./


# RUN ["chown", "-R", "ml-api-user:ml-api-user", "./"]

# ADD run.sh ./
# RUN ["chmod", "+x", "/Users/xinyu/repos/Fetch/ML/api/run.sh"]
# ENTRYPOINT ["/Users/xinyu/repos/Fetch/ML/API/run.sh"]


USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]
# ENV PATH=“${PATH}:/root/.local/bin”

# CMD ["bash", "./run.sh"]

# ENTRYPOINT []
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]