
## Objectives

You are expected to build an ML model from scratch to address this challenge. Your solution can be simple or complex. You are allowed to develop your solution using any languages and frameworks like PyTorch or Tensorflow. But please note that we would like to use your solution to understand your ML knowledge base. So please avoid from using any high level libraries like scikit-learn which makes it impossible to exhibit your ML skills.
Additionally, you are expected to build up a small app which will run an inference procedure against your own trained model and return the predicted results. You are free to build up any form of app like a web service or so but having user interaction and some sort of visualization will be a plus.
If possible, please package your app in a Docker container that can be built and run locally or pulled down and run via Docker Hub.
Please assume the evaluator does not have prior experience executing programs in your chosen language. Therefore, please include any documentation necessary to accomplish the above requirements.
The code, at a minimum, must run. Please provide clear instructions on how to run it.
When complete, please upload your codebase to a public Git repo (GitHub, Bitbucket, etc.) and email us the link. Please double-check this is publicly accessible.


## About

The current app supports the selection between the two different choices of prediction model (LSTM, GRU).

## Installation & Usage 

After Cloning the repo, follow the following steps

`cd ML`
`cd packages`
`cd ml_api`
`docker build -t api_image .`
`cd ../st`
`docker build -t st_image .`
`cd ..`
`docker-compose up -d --build`

Now you should be able to open your browser and navigate to http://localhost:8501 to use the application.