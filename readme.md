# Berdikari Bank API

This project is from IYKRA Data Fellowship Weekly Task to create an API that serves customer churning prediction.

## Description

The dataset is cleaned and exist in the `data` folder. We tried several machine learning algorithm and choose the best algorithm. Random Forest have the highest accuracy but the computation takes a lot of times. We switch to XGBoost for better performance in the API. Details about the analysis and modelling can be found in `analysis` folder.

![XGBoost performance](https://i.imgur.com/06EHk00.png "XGBoost performance")

We create the REST API services using Flask and tested the API in localhost and GCP's compute engine. The API require several parameter:

We send the data in JSON format as follow to the server.

```
{
    "CreditScore": 600,
    "Geography_cat": 0,
    "Gender_cat": 1,
    "Age": 35,
    "Tenure": 12,
    "Balance": 150000,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000
}
```

The result is:

!["Postman Screenshot"](https://i.imgur.com/VQxaFKL.png "Postman Screenshot")



## Description of Folders

* **analysis**: contains code for model training and jupyter notebook about the dataset analysis.
* **app**: contains code for the web application using flask to serves API.
* **data**: contains the dataset used for churning modelling.
* **models**: contains the pickled model to be used in the web application.

## How To Use Web Application Demo

1. Clone github repository to local computer
2. Download `Python 3`
3. Install required dependencies with the following command: `pip install -r requirements.txt`
4. Navigate to `app` folder
5. From command line, type: `python server.py`
6. Open web browser to "localhost:5000"
