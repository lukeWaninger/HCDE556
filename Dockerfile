FROM node:8.1.3

RUN mkdir -p /usr/src/app
WORKDIR /user/src/app
COPY . .

EXPOSE 8080

RUN npm install --silent

CMD ["npm", "start"]