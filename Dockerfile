FROM node:8.11.3

RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/src/app/src/data

WORKDIR /usr/src/app/src/data
RUN aws s3 cp s3://elasticbeanstalk-us-east-1-701856502070/RedditJokes/data/ data/ --recursive

WORKDIR /usr/src/app
COPY . .

ENV PATH /usr/src/app/node_modules/.bin:$PATH

EXPOSE 8080

COPY . /usr/src/app/
RUN npm install --silent

CMD ["npm", "start"]