# Building Product Recommendation Engine from Shopping Cart Data

In this project, a prototype recommendation engine was developed using shopping cart data available from Kaggle's [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?select=events.csv).  The gzipped version of the raw shopping "events" data file `events.csv.gz` can be found in the folder [`data`](data).  The file contains a list of three kinds of events: views, add-to-cart events, and transactions.

For each transaction, a list of product items purchased was extracted from the data file by running the script [`extract_shopping_cart_data.py`](extract_shopping_cart_data.py).  An exploratory analysis and visualizations of the extracted information is presented in [`EDA.ipynb`](EDA.ipynb).  Next, a machine learning model was trained for the following prediction task (see [`training.ipynb`](training.ipynb)): given the day of month, day of week, and hour of day of the transaction as well as the item(s) the customer already has placed in the shopping cart, predict for each product item the probability that the customer later will add it to the cart and purchase it.  A set of logistic regression models, with one model per product item, was used.

* create a post-processing logic to generate product recommendations
* provide analysis of the engine's predictive accuracy and characteristics


The modeling formulation is as follows: 


TODO: Include mentions/links for :
* 
* 
* trained_model.pickle
*	full_train_data.pickle.gz
* training.ipynb
* cleaned_data
* hit_rate.ipynb
* my_utilities.py
