FROM node:8.1.3

RUN mkdir -p /src
WORKDIR /src
COPY . .

EXPOSE 8080

RUN npm install --silent
RUN npm run build
RUN npm run start
