FROM python:3.10
# /Users/xinyu/repos/Fetch/ML/packages/api


WORKDIR /opt/st

ADD ./* /opt/st
# ENV PATH /opt/homebrew/bin:${PATH}
RUN pip install --upgrade pip
# RUN pip freeze > /Users/xinyu/repos/Fetch/ML/packages/st/requirements.txt
# RUN pip install -r /Users/xinyu/repos/Fetch/ML/packages/st/requirements.txt
RUN pip install --ignore-installed streamlit==1.12.0

EXPOSE 8501
# ENV PATH=“${PATH}:/opt/homebrew/bin”

ENTRYPOINT ["streamlit", "run"]

CMD ["./app.py"]