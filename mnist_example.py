
#%%
print('startup')
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Softmax, Flatten
#%%
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)

#%%
# Select 4 indeces of images of X_train to display
images = [0,1,2,3]

print('Here are some example images from the MNIST dataset:')
plt.subplot(221)
plt.title('image number: {}'.format(images[0]))
plt.imshow(X_train[images[0]])
plt.axis('off')
plt.subplot(222)
plt.title('image number: {}'.format(images[1]))
plt.imshow(X_train[images[1]])
plt.axis('off')
plt.subplot(223)
plt.title('image number: {}'.format(images[2]))
plt.imshow(X_train[images[2]])
plt.subplot(224)
plt.axis('off')
plt.title('image number: {}'.format(images[3]))
plt.imshow(X_train[images[3]])
plt.axis('off')
plt.show()
#%%

# Standardise the images
X_test = X_test / 255.0
X_train = X_train / 255.0

num_pixels = X_train.shape[1]*X_train.shape[2]
print('number of pixels per image = {}'.format(num_pixels))
#%%
num_classes = np.max(np.max(Y_train))+1
print('number of classes = {}'.format(num_classes))

#%%
# making the model
model = Sequential()
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(num_classes,activation = 'softmax'))

# compiling the model
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',
metrics = ['accuracy'])

#%%

# training the network
model.fit(x = X_train, y = Y_train, 
batch_size= 16, epochs = 5, )


#%%
print('Evaluation with test set:')
score = model.evaluate(x = X_test, y = Y_test, batch_size=16)
#%%

print('Loss = {:4.2f}'.format(score[0]))
print('Accuracy = {:4.2f}%'.format(score[1]*100))
#%%
test_print = [8737, 883, 937, 12]
prediction_probs = model.predict(X_test)
prediction = np.argmax(prediction_probs, axis=1)

#%%
print('Here are examples of predictions:')
test_print = [0, 1, 2, 3]
plt.subplot(221)
plt.title('Prediction: {}'.format(prediction[test_print[0]]))
plt.imshow(X_test[test_print[0]])
plt.axis('off')
plt.subplot(222)
plt.title('Prediction: {}'.format(prediction[test_print[1]]))
plt.imshow(X_test[test_print[1]])
plt.axis('off')
plt.subplot(223)
plt.title('Prediction: {}'.format(prediction[test_print[2]]))
plt.imshow(X_test[test_print[2]])
plt.axis('off')
plt.subplot(224)
plt.title('Prediction: {}'.format(prediction[test_print[3]]))
plt.imshow(X_test[test_print[3]])
plt.axis('off')
plt.show()

#%%
