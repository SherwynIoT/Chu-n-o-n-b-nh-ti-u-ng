import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.metrics import recall_score
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam


# Load the dataset
df = pd.read_csv('Dataset_challenge.csv')
skinthickness = df['SkinThickness'].median()
insulin = df['Insulin'].median()

# Thay thế các giá trị 0 bằng trung vị tương ứng
df['SkinThickness'] = df['SkinThickness'].replace(0, skinthickness)
df['Insulin'] = df['Insulin'].replace(0, insulin)
scaler = StandardScaler()
data = pd.DataFrame(df)
columns_to_normalize = ['Glucose', 'BloodPressure','Insulin','SkinThickness', 'BMI','DiabetesPedigreeFunction','Age',"Pregnancies"]
data[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Split the dataset into training and testing sets
X_train = df.iloc[:595, :-1]
Y_train = df.iloc[:595, -1]
X_test = df.iloc[596:, :-1]
Y_test = df.iloc[596:, -1]


# Create the model

model = keras.Sequential()

dense_1=Dense(128, activation='relu', input_shape=(X_train.shape[1],))
model.add(dense_1)
model.add(Dropout(0.5))

dense_2=Dense(64, activation='relu')
model.add(dense_2)
model.add(Dropout(0.5))

dense_3=Dense(32, activation='relu')
model.add(dense_3)
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)

# Compile the model

model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])


# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
Y_pred = (model.predict(X_test) > 0.5).astype("int32")
recall = recall_score(Y_test, Y_pred)
print('Accuracy:', accuracy)
print('Loss:', loss)
print('Recall:', recall)




# plot loss và accuracy
plt.figure(figsize=(20,10))
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
