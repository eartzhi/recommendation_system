FROM python:3.11.13

LABEL "maintainer"="eartzhi" \
      "app"="recomendation_server" \
      "version"="1.0"

COPY . /app
WORKDIR /app

ENV SERVER_USER="shop"
ENV PASSWORD="password"
ENV RES_PASSWORD=123

RUN pip install --upgrade pip
RUN pip install -r requirements.txt



ENTRYPOINT ["python"]
CMD ["server.py" ]

# EXPOSE 8000
EXPOSE 5000

VOLUME ./data:app/data

