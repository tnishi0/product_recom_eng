# Building Product Recommendation Engine from Shopping Cart Data

The goal of this project is to develop a prototype recommendation engine using a shopping cart data set.  To do this I will:
* train a machine learning model to predict purchases based on shopping cart status
* create a post-processing logic to generate product recommendations
* provide analysis of the engine's predictive accuracy and characteristics

I downloaded the raw shopping "events" data file `events.csv` from Kaggle's [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?select=events.csv).  The gzipped version of that file is in [the data folder](data).  The file contains a list of three kinds of events: views, add-to-cart events, and transactions.

The modeling formulation is as follows: given the transaction’s day of month, day of week, hour of day, and the item(s) the customer already has in the shopping cart, predict for each product item the probability that the customer later adds it to the cart and purchases it.

