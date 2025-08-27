FROM python:3.12

WORKDIR /usr/src/server/
COPY . .

ENV SERVER_USER=shop
ENV PASSWORD=password

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "server.py" ]

EXPOSE 8000
EXPOSE 5000

VOLUME .data .data

