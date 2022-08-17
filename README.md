# Building Product Recommendation Engine from Shopping Cart Data

In this project, a prototype recommendation engine was developed using shopping cart data available from Kaggle's [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?select=events.csv).  The gzipped version of the raw shopping "events" data file `events.csv.gz` can be found in the folder [`data`](data).  The file contains a list of three kinds of events: views, add-to-cart events, and transactions.

## Exloratory data analyis
For each transaction, a list of product items purchased was extracted from the data file by running the script [`extract_shopping_cart_data.py`](extract_shopping_cart_data.py).  The cleaned data was saved to files in the folder [`cleaned_data`](cleaned_data).  An exploratory analysis and visualizations of the extracted information is presented in [`EDA.ipynb`](EDA.ipynb).

## Training machine learning model
As a basis of product recommendation, the following prediction task was formulated: given the day of month, day of week, and hour of day of the transaction as well as the item(s) the customer already has placed in the shopping cart, predict for each product item the probability that the customer later will add it to the cart and purchase it.  A set of logistic regression models, one for each product item that the customer could purchase, were trained (see [`training.ipynb`](training.ipynb)).  The training was done using an expanded set of data, accounting for multiple time points during the customer's shopping experience (with different number of items already pleced in the cart).  This expanded data and the trained (set of) models are saved in [`full_train_data.pickle.gz`](full_train_data.pickle.gz) and [`trained_model.pickle`](trained_model.pickle), respectively.


* create a post-processing logic to generate product recommendations
Given a feature vector [encoding day of month, day of week, part of day,
channel, and first item(s)], the output of the model is a vector of predicted
probabilities, one for each product item
• After the customer adds the first item to the shopping cart, the engine selects
the 3 items with the highest probabilities
• After adding another item to the cart, the model is re-run and the
recommendations are updated
• Repeat after each additional item until the customer is done ordering
• Limited to 5 unique recommendations per order


* provide analysis of the engine's predictive accuracy and characteristics




TODO: Include mentions/links for :
* hit_rate.ipynb
* my_utilities.py
