"""Extracts the shopping cart data

Note: Running this on a MacBook Air with Apple M1 chip and 8GB memory
took ~2 min 36 sec.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
    
# num_items_to_select = 11183
num_items_to_select = 3000

# Load the "events" data file
file_name = 'data/events.csv.gz'
print(f'Loading "{file_name}"...')
df = pd.read_csv(file_name, compression='gzip')
print(f"Loaded {len(df)} rows of data")
print('Processing the data...')

# Create a new column with human-readble timestamps
df['datetime'] = pd.to_datetime(df.timestamp, unit='ms')

# Remove "view" events
df.drop(df.index[df['event'] == 'view'], inplace=True)

# List all unique transaction IDs
unique_transaction_id \
    = df[df['transactionid'].notnull()]['transactionid'].unique()
print(f'Found {len(unique_transaction_id)} unique transactions')

# Define a function to create an ordered list of purchased items for
# a give transaction
def get_ordered_items(tid):

    # Get the visitor associated with this transaction.  Throw error if
    # multiple visitors are associated with this transactions.
    transactions = df[df['transactionid'] == tid]
    visitor = transactions['visitorid'].unique()
    assert(len(visitor) == 1)
    visitor = visitor[0]

    # List the items purchased under this transaction ID.  There is one
    # item that is purchased multiple times in the same transaction by
    # the same visitor (itemid=285758, transactionid=5708.0,
    # visitorid=1384800).
    items = transactions['itemid'].values
    transaction_datetime = transactions['datetime'].values

    # Find the last transaction for each of these items
    add_to_cart_events = pd.DataFrame()
    for item, datetime in zip(items, transaction_datetime):
        all_add_to_cart_events = df[
            (df['event'] == 'addtocart') &
            (df['itemid'] == item) &
            (df['visitorid'] == visitor) &
            (df['datetime'] <= datetime)
        ].sort_values(by='datetime')
        if len(all_add_to_cart_events) > 0:
            add_to_cart_events = pd.concat(
                [add_to_cart_events, all_add_to_cart_events.iloc[[-1]]]
            )

    # Return the ordered comma-separated list of items (if the list is
    # not empty)
    if len(add_to_cart_events) > 0:
        ordered_itmes = add_to_cart_events[['datetime', 'itemid']] \
                        .sort_values(by='datetime')['itemid'] \
                        .transform(lambda x: str(x)) \
                        .values
        datetime = add_to_cart_events['datetime'].min()
        return datetime, ordered_itmes
    else:
        return None

# Apply the function above to create a pandas DataFrame with several
# features, including the ordered list of purchased items.
print('Building item list for each transaction...')
unique_transaction_id.sort()
df_output = pd.DataFrame(
    unique_transaction_id, columns=['tid'], dtype='int'
)
# df_output['tmp'] = df_output['tid'].apply(get_ordered_items)
df_output['tmp'] = df_output['tid'].parallel_apply(get_ordered_items)
df_output.dropna(inplace=True)
df_output['month'] = [item[0].month for item in df_output['tmp']]
df_output['day_of_month'] = [item[0].day for item in df_output['tmp']]
df_output['day_of_week'] = [
    item[0].dayofweek for item in df_output['tmp']]
df_output['datetime'] = [item[0] for item in df_output['tmp']]
df_output['hour'] = [item[0].hour for item in df_output['tmp']]
df_output['items'] = [item[1] for item in df_output['tmp']]
df_output.drop(labels='tmp', axis=1, inplace=True)
print('')

# Build a list of unique items.  This is different from the 417,053
# unique items mentioned in the dataset description, since all the
# "view" events were excluded.
purchases = df_output['items'] \
            .transform(lambda items: np.unique(items)) \
            .explode().values
column_name_mapping = {'index': 'item_id', 0: 'frequency'}
purchase_counts = pd.value_counts(purchases) \
                    .reset_index() \
                    .rename(columns=column_name_mapping) \
                    .sort_values(by='frequency', ascending=False)
unique_items = purchase_counts['item_id'].values
print(f"Selected {num_items_to_select} most frequently purchased",
      f"items out of {len(unique_items)} unique items found")
unique_items = unique_items[:num_items_to_select]
unique_items.sort()

# Remove items that are not in unique_items
df_output['items'] = df_output['items'].transform(
    lambda items: [item for item in items if item in unique_items]
)
df_output['items'] = df_output['items'].transform(
    lambda items: None if len(items) == 0 else items
)
df_output.dropna(inplace=True)
print(df_output.head())
print(f"Output DataFrame has {len(df_output)} rows")
print('First few rows:')
print(df_output.head())

# Split into training and test data and save them separately
# (training data: 90%, test data: 10%)
df_train, df_test = train_test_split(
    df_output, test_size=0.1, random_state=1)
data = {'df': df_train, 'unique_items': unique_items}
file_name = 'cleaned_data/train_data.pickle'
with open(file_name, 'wb') as f:
    pickle.dump(data, f)
print(f'Saved the training data to "{file_name}"')
data = {'df': df_test, 'unique_items': unique_items}
file_name = 'cleaned_data/test_data.pickle'
with open(file_name, 'wb') as f:
    pickle.dump(data, f)
print(f'Saved the test data to "{file_name}"')