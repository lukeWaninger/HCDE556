FROM node:8.1.3

RUN mkdir -p /usr/src/app
WORKDIR /user/src/app
COPY . .

ENV PATH /usr/src/app/node_modules/.bin:$PATH

EXPOSE 8080

COPY . /usr/src/app/
COPY /home/ec2-user/ /usr/src/app/src/data
RUN npm install --silent

CMD ["npm", "start"]