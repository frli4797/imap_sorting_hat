FROM python:3
ENV ish_path /opt/ish
ENV ISH_CONFIG_PATH /opt/ish/config

RUN apt-get update && apt-get install build-essential

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt ./
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY *.py ${ish_path}/ 
WORKDIR ${ish_path}
RUN ["mkdir", "-p", "${ISH_CONFIG_PATH}"]
CMD [ "python3", "ish.py" ]