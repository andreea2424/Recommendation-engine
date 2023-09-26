from collections import defaultdict
import pickle
import pandas as pd
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
# Load your data
df = pd.read_csv('/Users/andreeastanciu/Documents/csv/AL_DOUAZECEALEA_DATASET_final_dataset.csv')

# Keep only the necessary columns
df = df[['reviewer_id', 'listing_id', 'sentiment', 'price', 'latitude', 'longitude', 'host_is_superhost', 
         'number_of_reviews', 'accommodates', 'minimum_nights', 'maximum_nights']]


scaler = MinMaxScaler()
df['sentiment'] = scaler.fit_transform(df[['sentiment']])

# Convert host_is_superhost to numeric (1 for true, 0 for false)
df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})

# Scale continuous variables

scaler = StandardScaler()
df[['price', 'latitude', 'longitude', 'number_of_reviews', 'accommodates', 'minimum_nights', 'maximum_nights']]= scaler.fit_transform(df[['price', 'latitude', 'longitude', 'number_of_reviews', 'accommodates', 'minimum_nights', 'maximum_nights']])



# Calculate user-specific mean values
df['average_user_price'] = df.groupby('reviewer_id')['price'].transform('mean')
df['average_user_latitude'] = df.groupby('reviewer_id')['latitude'].transform('mean')
df['average_user_longitude'] = df.groupby('reviewer_id')['longitude'].transform('mean')
df['average_user_reviews'] = df.groupby('reviewer_id')['number_of_reviews'].transform('mean')
df['average_user_accommodates'] = df.groupby('reviewer_id')['accommodates'].transform('mean')

# Convert IDs into integer indices for embedding layer


user_enc = LabelEncoder()
df['user'] = user_enc.fit_transform(df['reviewer_id'].values)
user_mapping = user_enc.classes_


# Save the mapping
with open('user_mapping.pkl', 'wb') as f:
    pickle.dump(user_mapping, f)

item_enc = LabelEncoder()
df['item'] = item_enc.fit_transform(df['listing_id'].values)
item_mapping = item_enc.classes_
# Save the mapping
with open('item_mapping.pkl', 'wb') as f:
    pickle.dump(item_mapping, f)

n_users = df['user'].nunique()
n_items = df['item'].nunique()

# Split the data into train and test set
train, test = train_test_split(df, test_size=0.2, random_state=42)
# further split the train set into train and validation set
train, val = train_test_split(train, test_size=0.2, random_state=42)


# 2. Save these splits to disk
train.to_csv('train_dataset.csv', index=False)
test.to_csv('test_dataset.csv', index=False)
val.to_csv('val_dataset.csv', index=False)

# Now, whenever you start your process or in another script
# 3. Load these splits
train = pd.read_csv('train_dataset.csv')
test = pd.read_csv('test_dataset.csv')
val = pd.read_csv('val_dataset.csv')

# Hyperparameters
n_latent_factors = 10#inainte era 20, l am scazut la 17 ca sa vad daca mai e overfitted, daca il scazi la 17 nu e bine, l am  crescut la 22
early_stopping = EarlyStopping(monitor='val_loss', patience=2.5)#daca maresti numarul nu e bine, modelul devine overfitted

# User embedding path
user_input = Input(shape=[1], name='User')


user_embedding = Embedding(n_users, n_latent_factors, name='UserEmbedding')(user_input)


user_vec = Flatten(name='FlattenUsers')(user_embedding)

# Item embedding path
item_input = Input(shape=[1], name='Item')
item_embedding = Embedding(n_items, n_latent_factors, name='ItemEmbedding')(item_input)
item_vec = Flatten(name='FlattenItems')(item_embedding)

# Additional features

price_input = Input(shape=[1], name='Price')
latitude_input = Input(shape=[1], name='Latitude')
longitude_input = Input(shape=[1], name='Longitude')
reviews_input = Input(shape=[1], name='Reviews')
# accommodates_input = Input(shape=[1], name='Accommodates')
# min_nights_input = Input(shape=[1], name='MinNights')
# max_nights_input = Input(shape=[1], name='MaxNights')




# Concatenate everything together



concat = Concatenate()([user_vec, item_vec, price_input, latitude_input, longitude_input, 
                        reviews_input])




# Add a Dropout layer after the concatenation


concat = Dropout(0.1)(concat)

# Dense layer to learn non-linear interactions
dense = Dense(5, activation='relu', kernel_regularizer=l2(0.001))(concat) # L2 regularization added here

# Add a Dropout layer after the Dense layer


dense = Dropout(0.1)(dense)

# Output layer
rating = Dense(1, activation='tanh', kernel_regularizer=l2(0.001))(dense) 




# L2 regularization added here # am scazut numarul de neuroni de la 1 la 0.5


# Define the model


model = Model([user_input, item_input, price_input, latitude_input, longitude_input, reviews_input], 
              rating)

learning_rate=0.0001179


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])




# Train the model
history = model.fit(
    [train.user, train.item, train.average_user_price, train.average_user_latitude, 
                             train.average_user_longitude, train.average_user_reviews
                             ], 
    train.sentiment, 
    validation_data=([val.user, val.item, val.average_user_price, val.average_user_latitude, 
                      val.average_user_longitude, val.average_user_reviews], val.sentiment),
    epochs=22, 
    verbose=1,
    callbacks=[early_stopping]
)
# Save the hyperparameters as strings
n_latent_factors_str = str(n_latent_factors)
early_stopping_patience_str = str(3)
epochs_str = str(25)
dense_str= str(5)


# Save the model
model.save('enhanced_recommender.h5')
specific_user_id = 24620243
specific_user_id2 = 392358593
print(specific_user_id)

user_data = test[test['reviewer_id'] == specific_user_id]
user_predictions = model.predict([user_data.user, user_data.item, user_data.price, 
                                  user_data.latitude, user_data.longitude, user_data.number_of_reviews, 
                                  ])

print(user_predictions)
# Make predictions
predictions = model.predict([test.user, test.item, test.average_user_price, test.average_user_latitude, 
                             test.average_user_longitude, test.average_user_reviews])

# Print the predictions
print(predictions)

# Given the specific user ID, get its encoded value
specific_user_encoded = user_enc.transform([specific_user_id])[0]

specific_user = user_enc.transform([specific_user_id])[0]
all_items = np.array(list(set(df.item)))
# Create a dataframe for the specific user with all the items
user_dataset = pd.DataFrame({
    'user': [specific_user]*len(all_items),
    'item': all_items
})
# Append other features: e.g., average price, average latitude, etc. 
# Here I am using the averages for the specific user, but you can modify as required.
user_dataset['price'] = df.loc[df['user'] == specific_user, 'average_user_price'].iloc[0]
user_dataset['latitude'] = df.loc[df['user'] == specific_user, 'average_user_latitude'].iloc[0]
user_dataset['longitude'] = df.loc[df['user'] == specific_user, 'average_user_longitude'].iloc[0]
user_dataset['number_of_reviews'] = df.loc[df['user'] == specific_user, 'average_user_reviews'].iloc[0]
user_dataset['accommodates'] = df.loc[df['user'] == specific_user, 'average_user_accommodates'].iloc[0]
user_dataset['minimum_nights'] = df['minimum_nights'].mean()  # global average
user_dataset['maximum_nights'] = df['maximum_nights'].mean()  # global average
user_all_predictions = model.predict([
    user_dataset.user, user_dataset.item, user_dataset.price, 
    user_dataset.latitude, user_dataset.longitude, 
    user_dataset.number_of_reviews
])
user_dataset['predicted_sentiment'] = user_all_predictions
top_10_items = user_dataset.sort_values(by='predicted_sentiment', ascending=False).head(10)

# Decode the item IDs to the original listing IDs
top_10_listing_ids = item_enc.inverse_transform(top_10_items['item'])

print("Top 10 recommended listings for user {}:".format(specific_user_id))
print(user_all_predictions)
print(top_10_listing_ids)


##########
specific_user_encoded = user_enc.transform([specific_user_id2])[0]

specific_user = user_enc.transform([specific_user_id2])[0]
all_items = np.array(list(set(df.item)))
# Create a dataframe for the specific user with all the items
user_dataset = pd.DataFrame({
    'user': [specific_user]*len(all_items),
    'item': all_items
})
# Append other features: e.g., average price, average latitude, etc. 
# Here I am using the averages for the specific user, but you can modify as required.
user_dataset['price'] = df.loc[df['user'] == specific_user, 'average_user_price'].iloc[0]
user_dataset['latitude'] = df.loc[df['user'] == specific_user, 'average_user_latitude'].iloc[0]
user_dataset['longitude'] = df.loc[df['user'] == specific_user, 'average_user_longitude'].iloc[0]
user_dataset['number_of_reviews'] = df.loc[df['user'] == specific_user, 'average_user_reviews'].iloc[0]
user_dataset['accommodates'] = df.loc[df['user'] == specific_user, 'average_user_accommodates'].iloc[0]
user_dataset['minimum_nights'] = df['minimum_nights'].mean()  # global average
user_dataset['maximum_nights'] = df['maximum_nights'].mean()  # global average
user_all_predictions = model.predict([
    user_dataset.user, user_dataset.item, user_dataset.price, 
    user_dataset.latitude, user_dataset.longitude, 
    user_dataset.number_of_reviews
])
user_dataset['predicted_sentiment'] = user_all_predictions
top_10_items = user_dataset.sort_values(by='predicted_sentiment', ascending=False).head(10)

# Decode the item IDs to the original listing IDs
top_10_listing_ids = item_enc.inverse_transform(top_10_items['item'])

print("Top 10 recommended listings for user {}:".format(specific_user_id2))
print(user_all_predictions)
print(top_10_listing_ids)


plt.figure(figsize=(10,5))
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(f'Model_MAE_factors_option1_{n_latent_factors_str}_patience_{early_stopping_patience_str}_epochs_{epochs_str}_dense_{dense_str}.png')
plt.close()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(f'Model_Loss_factors_option10_patiencce_3{n_latent_factors_str}_patience_{early_stopping_patience_str}_epochs_{epochs_str}_dense_{dense_str}.png')
plt.close()
# # Evaluation
mse = mean_squared_error(test.sentiment.values, predictions.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(test.sentiment.values, predictions.flatten())

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# # Load the user ID mappings
# with open('user_mapping.pkl', 'rb') as f:
#     user_mapping = pickle.load(f)
# import matplotlib.pyplot as plt

# # assuming history is the output of your model.fit()

# Plotting RMSE
plt.figure(figsize=(10, 5))
plt.plot(np.sqrt(history.history['loss']), label='Training RMSE')
plt.plot(np.sqrt(history.history['val_loss']), label='Validation RMSE')
plt.title('RMSE over epochs')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(f'Model_RMSE.png')
plt.show()

# Plotting MSE
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.title('MSE over epochs')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(f'Model_MSE.png')
plt.show()

# # # Find the user index for the specific user ID
# # user_index = np.where(user_mapping == 178267981)[0][0]

# # # Fetch user-specific values from the dataframe
# # user_specific_values = df[df['reviewer_id'] == 178267981].iloc[0]

# # # Create the input arrays for the specific user
# # user_indices = np.full(n_items, user_index)
# # item_indices = np.arange(n_items)  
# # price_array = np.full(n_items, user_specific_values['average_user_price'])
# # latitude_array = np.full(n_items, user_specific_values['average_user_latitude'])
# # longitude_array = np.full(n_items, user_specific_values['average_user_longitude'])
# # reviews_array = np.full(n_items, user_specific_values['average_user_reviews'])

# # # Reshape the input arrays to match the model's input shape
# # user_indices = user_indices.reshape((-1, 1))
# # item_indices = item_indices.reshape((-1, 1))
# # price_array = price_array.reshape((-1, 1))
# # latitude_array = latitude_array.reshape((-1, 1))
# # longitude_array = longitude_array.reshape((-1, 1))
# # reviews_array = reviews_array.reshape((-1, 1))

# # # Make predictions using the model for the specific user
# # predictions = model.predict([user_indices, item_indices, price_array, latitude_array, longitude_array, reviews_array])

# # # Get the indices of the top 5 recommended items for the specific user
# # top_5_item_indices = predictions.flatten().argsort()[-5:][::-1]

# # # Convert item indices to item IDs
# # top_5_item_ids = item_mapping[top_5_item_indices]

# # # Print the top 5 recommended item IDs for the user
# # for item_id in top_5_item_ids:
# #     print(item_id)




# # Assuming that 'recommendations' is a dictionary 
# # where keys are user_ids and values are lists of recommended item_ids
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict

# # Assuming that 'recommendations' is a dictionary 
# # where keys are user_ids and values are lists of recommended item_ids
# recommendations = defaultdict(list) 
# # Fill 'recommendations' with your actual recommendations

# # Calculate average cosine similarity for each pair of users
# average_similarity = np.mean([
#     cosine_similarity(
#         [recommendations[user1]], 
#         [recommendations[user2]]
#     ) 
#     for user1 in recommendations.keys() 
#     for user2 in recommendations.keys() 
#     if user1 != user2
# ])

# # Since cosine similarity ranges from -1 to 1, 
# # subtracting the average similarity from 1 gives a measure of personalization
# personalization = 1 - average_similarity
# print('Personalization:', personalization)

# # Calculate RMSE from MSE
# rmse = np.sqrt(history.history['loss'])
# val_rmse = np.sqrt(history.history['val_loss'])

# # Generate the plot
# plt.figure(figsize=(10, 6))
# plt.plot(rmse, label='Training RMSE')
# plt.plot(val_rmse, label='Validation RMSE')
# plt.title('RMSE over epochs')
# plt.xlabel('Epoch')
# plt.ylabel('RMSE')
# plt.legend()

# plt.show()

# # # Extract user embeddings
# # user_embeddings = model.get_layer('UserEmbedding').get_weights()[0]

# # # Extract item embeddings
# # item_embeddings = model.get_layer('ItemEmbedding').get_weights()[0]
# # from sklearn.manifold import TSNE

# # # Apply t-SNE to user embeddings
# # user_tsne = TSNE(n_components=2).fit_transform(user_embeddings)

# # # Apply t-SNE to item embeddings
# # item_tsne = TSNE(n_components=2).fit_transform(item_embeddings)
# # import matplotlib.pyplot as plt

# # plt.figure(figsize=(10,5))
# # plt.scatter(user_tsne[:, 0], user_tsne[:, 1], label='Users')
# # plt.scatter(item_tsne[:, 0], item_tsne[:, 1], label='Items')
# # plt.xlabel('Dimension 1')
# # plt.ylabel('Dimension 2')
# # plt.legend()
# # plt.title('t-SNE Visualization of Embeddings')
# # plt.savefig('t-SNE_Visualization_of_Embeddings.png')
# # plt.show()  # This line will display the plot


# # from sklearn.manifold import TSNE

# # # Apply t-SNE to user embeddings
# # user_tsne = TSNE(n_components=2).fit_transform(user_embeddings)

# # import matplotlib.pyplot as plt

# # plt.figure(figsize=(10,5))
# # plt.scatter(user_tsne[:, 0], user_tsne[:, 1], label='Users')
# # plt.xlabel('Dimension 1')
# # plt.ylabel('Dimension 2')
# # plt.legend()
# # plt.title('t-SNE Visualization of User Embeddings')
# # plt.savefig('t-SNE_Visualization_of_User_Embeddings.png')
# # plt.show()  # This line will display the plot

# # # Apply t-SNE to item embeddings
# # item_tsne = TSNE(n_components=2).fit_transform(item_embeddings)

# # plt.figure(figsize=(10,5))
# # plt.scatter(item_tsne[:, 0], item_tsne[:, 1], label='Items')
# # plt.xlabel('Dimension 1')
# # plt.ylabel('Dimension 2')
# # plt.legend()
# # plt.title('t-SNE Visualization of Item Embeddings')
# # plt.savefig('t-SNE_Visualization_of_Item_Embeddings.png')
# # plt.show()  # This line will display the plot
# # import shap
# # # Get the required columns from the train data
# # explainer_data = [
# #     train['user'].values, 
# #     train['item'].values, 
# #     train['average_user_price'].values, 
# #     train['average_user_latitude'].values, 
# #     train['average_user_longitude'].values, 
# #     train['average_user_reviews'].values, 
# #     train['accommodates'].values, 
# #     train['minimum_nights'].values, 
# #     train['maximum_nights'].values
# # ]
# # if isinstance(explainer_data, list):
# #     explainer_data = np.array(explainer_data)
# # # Pass the correctly formatted data to the explainer
# # explainer = shap.KernelExplainer(model.predict, explainer_data) # train_data should be the input training data you used

# # # Example prediction input for SHAP explanation
# # prediction_input = [test.user.iloc[:1], test.item.iloc[:1], test.average_user_price.iloc[:1], 
# #                     test.average_user_latitude.iloc[:1], test.average_user_longitude.iloc[:1], 
# #                     test.average_user_reviews.iloc[:1], test.accommodates.iloc[:1], 
# #                     test.minimum_nights.iloc[:1], test.maximum_nights.iloc[:1]]

# # # Compute SHAP values for the specific prediction
# # shap_values = explainer.shap_values(prediction_input)

# # # Plot the SHAP values
# # shap.summary_plot(shap_values, prediction_input)

# # # To save the plot
# # plt.savefig('shap_plot.png')

# # 1. Extract Item Embeddings
# item_embedding_layer = model.get_layer('ItemEmbedding')
# item_embeddings = item_embedding_layer.get_weights()[0]

# # 2. Compute Similarity Matrix
# item_similarity = cosine_similarity(item_embeddings)

# # 3. Visualize Similarity Matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# subset_size = 50 # Adjust this based on your available memory
# subset_similarity = item_similarity[:subset_size, :subset_size]
# subset_labels = [item_mapping[i] for i in range(subset_size)]

# plt.figure(figsize=(12, 12))
# sns.heatmap(subset_similarity, annot=False, cmap="PuRd", xticklabels=subset_labels, yticklabels=subset_labels)
# plt.title("Item-Item Similarity Matrix (Subset)")
# plt.xlabel("Item ID")
# plt.ylabel("Item ID")
# plt.show()

