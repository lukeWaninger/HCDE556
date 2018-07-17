FROM node:8.1.3

RUN mkdir -p /usr/src/api
WORKDIR /usr/src/api
COPY . .

EXPOSE 8080

CMD ["node", "."]