FROM node:6.9.5

RUN mkdir -p /usr/src/api
WORKDIR /usr/src/api
COPY . .

CMD ["node", "."]