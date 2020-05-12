import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import metrics



class dataloader():
    def __init__(self, 
                 coeff = 1,
                 batch_size=20):
        self.batch_size = batch_size
        self.N = 2000
        self.coeff = coeff
        self._reset()

    def __iter__(self):
        return self

    def __len__(self):
        #N = len(self.dataset)
        N = self.N
        b = self.batch_size
        return N // b + bool(N % b)

    def _reset(self):
        self._idx = 0

    def __next__(self):
        if self._idx >= self.N:
            self._reset()
            raise StopIteration()

        ### collocation points and governing equations
        x = (5 * np.random.rand(self.batch_size))[:, np.newaxis].astype(np.float32)
        y = self.coeff * x
        ###

        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        self._idx += self.batch_size
        return x, y


def encoder():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(1, activation=None, input_shape=(1,))
    ])
    return model



def train_model(model, data, path2save):

    epoch = 100
    #criterion = tf.losses.MeanAbsoluteError()
    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = optimizers.Adam(learning_rate=1e-2)
    train_loss = metrics.Mean()

    def compute_loss(y, preds):
        loss = criterion(y, preds)
        return loss

    @tf.function
    def train_step(mdl, x, y):
        with tf.GradientTape() as tape:
            preds = mdl(x)
            loss = compute_loss(y, preds)
        grads = tape.gradient(loss, mdl.trainable_variables)
        optimizer.apply_gradients(zip(grads, mdl.trainable_variables))
        train_loss(loss)
    
    for epoch in range(1, epoch + 1):
        for (x, y) in data:
            train_step(model, x, y)

        print('Epoch: {}, Cost: {:.3f}'.format(
            epoch+1,
            train_loss.result()
        ))
    #model.save_weights(path2save)
    model.save(path2save + '.h5')
    print('saved')


def make_dir(dir_name):
    if os.path.exists(dir_name) == False:
        os.mkdir(dir_name)



def create_model():

    #
    model_1 = encoder()
    path2save_1 = './model1'
    data_1 = dataloader(1, 20)
    train_model(model_1, data_1, path2save_1)

    model_2 = encoder()
    path2save_2 = './model2'
    data_2 = dataloader(2, 20)
    train_model(model_2, data_2, path2save_2)
    


def load():

    mse1 = metrics.Mean()
    mse2 = metrics.Mean()
    def compute_loss1(y, preds):
        loss = criterion(y, preds)
        mse1(loss)
    def compute_loss2(y, preds):
        loss = criterion(y, preds)
        mse2(loss)

    model_1 = encoder()
    path2save_1 = './model1'
    model_1 = keras.models.load_model(path2save_1 + '.h5')
    model_1.summary()

    #criterion = tf.losses.MeanAbsoluteError()
    criterion = tf.keras.losses.MeanSquaredError()

    

    for (x, y) in dataloader(1, 20):
        pred = model_1.predict(x)
        compute_loss1(y, pred)
    print('loss for model1', mse1.result())

    #model_2 = encoder()
    path2save_2 = './model2'
    model_2 = keras.models.load_model(path2save_2 + '.h5')
    model_2.summary()

    for (x, y) in dataloader(2, 20):
        pred = model_2.predict(x)
        compute_loss2(y, pred)
    print('loss for model2', mse2.result())

if __name__ == '__main__':

    create_model()

    load()

