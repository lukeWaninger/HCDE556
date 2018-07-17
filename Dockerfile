FROM node:8.11.3

RUN apt-get update -y
RUN apt-get install -y awscli

RUN mkdir -p /usr/src/app
RUN aws s3 cp s3://elasticbeanstalk-us-east-1-701856502070/RedditJokes/data/ /usr/src/app/src/data --recursive

WORKDIR /user/src/app
COPY . .

ENV PATH /usr/src/app/node_modules/.bin:$PATH

EXPOSE 8080

RUN npm install --silent
CMD ["npm", "start"]