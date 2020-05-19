import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from _dataloader import dataloader

class encoder(Model):
    def __init__(self, n_map=4):
        super().__init__()
        #self.l1 = Dense(1, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l1(1e-4))
        self.l1 = Dense(1, activation=None, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l1(1e-4))

    def call(self, x):

        h = self.l1(x)
        return h


def train_model(model, data, path2save):

    epoch = 100
    #criterion = tf.losses.MeanAbsoluteError()
    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = optimizers.Adam(learning_rate=1e-2)
    train_loss = metrics.Mean()

    def compute_loss(y, preds):
        #print(y.shape, preds.shape)
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
    model.save_weights(path2save)
    print('saved')


def make_dir(dir_name):
    if os.path.exists(dir_name) == False:
        os.mkdir(dir_name)


def create_model():
    make_dir('./checkpoints1')
    make_dir('./checkpoints2')

    #
    model_1 = encoder()
    path2save_1 = './checkpoints1/weight'
    data_1 = dataloader(1, 20)
    train_model(model_1, data_1, path2save_1)

    model_2 = encoder()
    path2save_2 = './checkpoints2/weight'
    data_2 = dataloader(2, 20)
    train_model(model_2, data_2, path2save_2)
    


def load():

    criterion = tf.keras.losses.MeanSquaredError()
    mse1 = metrics.Mean()
    mse2 = metrics.Mean()
    def compute_loss1(y, preds):
        loss = criterion(y, preds)
        mse1(loss)
    def compute_loss2(y, preds):
        loss = criterion(y, preds)
        mse2(loss)

    model_1 = encoder()
    path2save_1 = './checkpoints1/weight'
    model_1.load_weights(path2save_1)

    for (x, y) in dataloader(1, 20):
        pred = model_1.predict(x)
        compute_loss1(y, pred)
    print('loss for model1', mse1.result())

    model_2 = encoder()
    path2save_2 = './checkpoints2/weight'
    model_2.load_weights(path2save_2)

    for (x, y) in dataloader(2, 20):
        pred = model_2.predict(x)
        compute_loss2(y, pred)
    print('loss for model2', mse2.result())


def train_by_othermodel():

    model_1 = encoder()
    path2save_1 = './checkpoints1/weight'
    model_1.load_weights(path2save_1)
    model_2 = encoder()
    path2save_2 = './checkpoints2/weight'
    model_2.load_weights(path2save_2)
    model_1.l1.trainable = False
    model_2.l1.trainable = False

    model_3 = encoder()
    path2save_3 = './model3'
    
    epoch = 100
    optimizer = optimizers.Adam(learning_rate=1e-3)
    criterion = tf.keras.losses.MeanSquaredError()
    mse_train = metrics.Mean()
    mse_test = metrics.Mean()
    def compute_loss_train(y, preds):
        loss = criterion(y, preds)
        return loss
        
    def compute_loss_test(y, preds):
        loss = criterion(y, preds)
        mse_test(loss)

    @tf.function
    def train_step(mdl, x, y):
        with tf.GradientTape() as tape:
            preds = mdl(x)
            loss = compute_loss_train(y, preds)
        grads = tape.gradient(loss, mdl.trainable_variables)
        optimizer.apply_gradients(zip(grads, mdl.trainable_variables))
        mse_train(loss)
        
    # train loop
    for epoch in range(1, epoch + 1):
        for (x, _) in dataloader(0, 20):
            y = model_1.predict(x) + model_2.predict(x)
            y = tf.convert_to_tensor(y)
            train_step(model_3, x, y)

        print('Epoch: {}, Cost: {:.3f}'.format(
            epoch+1,
            mse_train.result()
        ))
    model_3.save_weights(path2save_3)
    print('saved')

    del model_1
    del model_2

    for (x, y) in dataloader(3, 20):
        pred = model_3.predict(x)
        compute_loss_test(y, pred)

    print('loss for model3', mse_test.result())


if __name__ == '__main__':

    #create_model()

    #load()

    train_by_othermodel()