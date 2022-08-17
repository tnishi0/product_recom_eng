# Building Product Recommendation Engine from Shopping Cart Data

In this project, a prototype recommendation engine was developed using shopping cart data available from Kaggle's [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?select=events.csv).  The gzipped version of the raw shopping "events" data file `events.csv` can be found in [the data folder](data).  The data file contains a list of three kinds of events: views, add-to-cart events, and transactions.

For each transaction, a list of product items purchased was extracted from the data file by running the script `extract_shopping_cart_data.py`.  An exploratory analysis of the extracted information is presented in [`EDA.ipynb`](EDA.ipynb).  
* train a machine learning model to predict purchases based on shopping cart status
* create a post-processing logic to generate product recommendations
* provide analysis of the engine's predictive accuracy and characteristics


The modeling formulation is as follows: given the transaction’s day of month, day of week, hour of day, and the item(s) the customer already has in the shopping cart, predict for each product item the probability that the customer later adds it to the cart and purchases it.


TODO: Include mentions/links for :
* 
* 
* trained_model.pickle
*	full_train_data.pickle.gz
* training.ipynb
* cleaned_data
* hit_rate.ipynb
* my_utilities.py
