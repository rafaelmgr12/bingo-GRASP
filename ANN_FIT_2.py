#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

test_tam = 0.3
realizations = 300



colunas = ['x','y','z','theta','phi']


# # Fit Z

data = pd.read_csv('data-optical-design (7).csv', usecols= colunas)




scaler1 = StandardScaler()
scaler2 = StandardScaler()
data[['x','y']] = scaler1.fit_transform(data[['x','y']])
data[['z']] = scaler2.fit_transform(data[['z']])
X_z = data[['x','y']]
y_z = data['z']


col_z = ['x','y']
colunas_z = [tf.feature_column.numeric_column(key = c) for c in col_z]





rep = realizations
for j in tqdm(range(rep)):
    colunas = ['x','y','z','theta','phi']
    data = pd.read_csv('data-optical-design (4).csv', usecols= colunas)


    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    data[['x','y']] = scaler1.fit_transform(data[['x','y']])
    data[['z']] = scaler2.fit_transform(data[['z']])
    X_z = data[['x','y']]
    y_z = data['z']


    X_train_z,X_test_z,y_train_z,y_test_z = train_test_split(X_z,y_z,test_size = test_tam)


    fn_z = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train_z, y = y_train_z, batch_size= 8,
                                                   num_epochs= None, shuffle= True)

    reg_z = tf.estimator.DNNRegressor(hidden_units=[10], feature_columns=colunas_z,
                                      activation_fn=tf.keras.activations.softmax,
                                      optimizer= lambda: tf.keras.optimizers.SGD(
            learning_rate=tf.compat.v1.train.exponential_decay(
                learning_rate=0.1,
                global_step=tf.compat.v1.train.get_global_step(),
                decay_steps=10000,
                decay_rate=0.96)))


    reg_z.train(input_fn = fn_z, steps = 20000)
    fn_prev_z = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_test_z, shuffle = False)

    val_prev_z= []
    for p in reg_z.predict(input_fn=fn_prev_z):
        val_prev_z.append(p['predictions'][0])

    aux = np.asarray(scaler2.inverse_transform(val_prev_z)).reshape(-1,1)
    np.save('data_output/data_ypre_test_z'+ str(j)+'.npy',aux)

    fn_prev_z = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train_z, shuffle = False)

    val_prev_z= []
    for p in reg_z.predict(input_fn=fn_prev_z):
        val_prev_z.append(p['predictions'][0])

    aux = np.asarray(scaler2.inverse_transform(val_prev_z)).reshape(-1,1)
    np.save('data_output/data_ypre_train_z'+ str(j)+'.npy',aux)

    fn_prev_z = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_z, shuffle = False)


    val_prev_z= []
    for p in reg_z.predict(input_fn=fn_prev_z):
        val_prev_z.append(p['predictions'][0])


    aux = np.asarray(scaler2.inverse_transform(val_prev_z)).reshape(-1,1)
    np.save('data_output/data_ypre_total_z'+ str(j)+'.npy',aux)

    X_train_z = scaler1.inverse_transform(X_train_z)
    y_train_z = scaler2.inverse_transform(y_train_z)
    X_test_z = scaler1.inverse_transform(X_test_z)
    y_test_z = scaler2.inverse_transform(y_test_z)


    np.save('data_output/data_xtrain_z'+ str(j)+'.npy',X_train_z)
    np.save('data_output/data_ytrain_z'+ str(j)+'.npy',y_train_z.reshape(-1,1))
    np.save('data_output/data_xtest_z'+ str(j)+'.npy',X_test_z)
    np.save('data_output/data_ytest_z'+ str(j)+'.npy',y_test_z.reshape(-1,1))

    #X_train_z.to_csv('data_output/data_xtrain_z'+ str(j)+'.csv')
    #y_train_z.to_csv('data_output/data_ytrain_z'+ str(j)+'.csv',header = True)
    #X_test_z.to_csv('data_output/data_xtest_z'+ str(j)+'.csv')
    #y_test_z.to_csv('data_output/data_ytest_z'+ str(j)+'.csv', header = True)


# # Fit $\theta$

data = pd.read_csv('data-optical-design (4).csv', usecols= colunas)



scaler3 = StandardScaler()
scaler4 = StandardScaler()
data[['x','y','z']] = scaler3.fit_transform(data[['x','y','z']])
data[['theta']] = scaler4.fit_transform(data[['theta']])
X_theta = data[['x','y','z']]
y_theta = data['theta']




col_theta = ['x','y','z']
colunas_theta = [tf.feature_column.numeric_column(key = c) for c in col_theta]


rep = realizations
for j in tqdm(range(rep)):

    colunas = ['x','y','z','theta','phi']
    data = pd.read_csv('data-optical-design (3).csv', usecols= colunas)

    scaler3 = StandardScaler()
    scaler4 = StandardScaler()
    data[['x','y','z']] = scaler3.fit_transform(data[['x','y','z']])
    data[['theta']] = scaler4.fit_transform(data[['theta']])
    X_theta = data[['x','y','z']]
    y_theta = data['theta']


    X_train_t,X_test_t,y_train_t,y_test_t = train_test_split(X_theta,y_theta, test_size = test_tam)


    fn_theta = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train_t, y = y_train_t, batch_size= 8,
                                                   num_epochs= None, shuffle= True)
    reg_theta = tf.estimator.DNNRegressor(hidden_units=[10], feature_columns=colunas_theta,
                                  activation_fn=tf.nn.relu, optimizer='Adam')
    reg_theta.train(input_fn = fn_theta, steps = 20000)

    fn_prev_theta = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_test_t, shuffle = False)

    val_prev_theta= []
    for p in reg_theta.predict(input_fn=fn_prev_theta):
        val_prev_theta.append(p['predictions'][0])
    aux = np.asarray(scaler4.inverse_transform(val_prev_theta)).reshape(-1,1)
    np.save('data_output/data_ypre_test_theta'+ str(j)+'.npy',aux)


    fn_prev_theta = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train_t, shuffle = False)

    val_prev_theta= []
    for p in reg_theta.predict(input_fn=fn_prev_theta):
        val_prev_theta.append(p['predictions'][0])
    aux = np.asarray(scaler4.inverse_transform(val_prev_theta)).reshape(-1,1)
    np.save('data_output/data_ypre_train_theta'+ str(j)+'.npy',aux)

    fn_prev_theta = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_theta, shuffle = False)

    val_prev_theta= []
    for p in reg_theta.predict(input_fn=fn_prev_theta):
        val_prev_theta.append(p['predictions'][0])
    aux = np.asarray(scaler4.inverse_transform(val_prev_theta)).reshape(-1,1)

    np.save('data_output/data_ypre_total_theta'+ str(j)+'.npy',aux)

    X_train_t = scaler3.inverse_transform(X_train_t)
    y_train_t = scaler4.inverse_transform(y_train_t)
    X_test_t = scaler3.inverse_transform(X_test_t)
    y_test_t = scaler4.inverse_transform(y_test_t)

    np.save('data_output/data_xtrain_theta'+ str(j)+'.npy',X_train_t)
    np.save('data_output/data_ytrain_theta'+ str(j)+'.npy',y_train_t.reshape(-1,1))
    np.save('data_output/data_xtest_theta'+ str(j)+'.npy',X_test_t)
    np.save('data_output/data_ytest_theta'+ str(j)+'.npy',y_test_t.reshape(-1,1))

    #X_train_t.to_csv('data_output/data_xtrain_theta'+ str(j)+'.csv')
    #y_train_t.to_csv('data_output/data_ytrain_theta'+ str(j)+'.csv',header = True)
    #X_test_t.to_csv('data_output/data_xtest_theta'+ str(j)+'.csv')
    #y_test_t.to_csv('data_output/data_ytest_theta'+ str(j)+'.csv', header = True)


## Fit phi



scaler5 = StandardScaler()
scaler6 = StandardScaler()
data[['x','y','theta']] = scaler5.fit_transform(data[['x','y','theta']])
data[['phi']] = scaler6.fit_transform(data[['phi']])
X_phi = data[['x','y','theta']]
y_phi = data['phi']


col_phi = ['x','y','theta']
colunas_phi = [tf.feature_column.numeric_column(key = c) for c in col_phi]

rep = realizations
for j in tqdm(range(rep)):

    colunas = ['x','y','z','theta','phi']
    data = pd.read_csv('data-optical-design (4).csv', usecols= colunas)

    scaler5 = StandardScaler()
    scaler6 = StandardScaler()
    data[['x','y','theta']] = scaler5.fit_transform(data[['x','y','theta']])
    data[['phi']] = scaler6.fit_transform(data[['phi']])
    X_phi = data[['x','y','theta']]
    y_phi = data['phi']


    X_train_phi,X_test_phi,y_train_phi,y_test_phi = train_test_split(X_phi,y_phi,test_size = test_tam)

    fn_phi = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train_phi, y = y_train_phi, batch_size= 8,
                                                   num_epochs= None, shuffle= True)
    reg_phi = tf.estimator.DNNRegressor(hidden_units=[10],
                                    feature_columns=colunas_phi,
                                    activation_fn=tf.keras.activations.tanh,
                                    optimizer= lambda: tf.keras.optimizers.SGD(
                                    learning_rate=
                                    tf.compat.v1.train.exponential_decay(
                                                            learning_rate=0.1,
                                global_step=tf.compat.v1.train.get_global_step(),
                                                                decay_steps=10000,
            decay_rate=0.96)))

    reg_phi.train(input_fn = fn_phi, steps = 20000)


    fn_prev_phi = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_test_phi, shuffle = False)


    val_prev_phi= []
    for p in reg_phi.predict(input_fn=fn_prev_phi):
        val_prev_phi.append(p['predictions'][0])
    aux = np.asarray(scaler6.inverse_transform(val_prev_phi)).reshape(-1,1)
    np.save('data_output/data_ypre_test_phi'+ str(j)+'.npy',aux)

    fn_prev_phi = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train_phi, shuffle = False)


    val_prev_phi= []
    for p in reg_phi.predict(input_fn=fn_prev_phi):
        val_prev_phi.append(p['predictions'][0])
    aux = np.asarray(scaler6.inverse_transform(val_prev_phi)).reshape(-1,1)
    np.save('data_output/data_ypre_train_phi'+ str(j)+'.npy',aux)

    fn_prev_phi = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_phi, shuffle = False)

    val_prev_phi = []
    for p in reg_phi.predict(input_fn=fn_prev_phi):
        val_prev_phi.append(p['predictions'][0])
    aux = np.asarray(scaler6.inverse_transform(val_prev_phi)).reshape(-1,1)

    np.save('data_output/data_ypre_total_phi'+ str(j)+'.npy',aux)


    X_train_phi = scaler5.inverse_transform(X_train_phi)
    y_train_phi = scaler6.inverse_transform(y_train_phi)
    X_test_phi = scaler5.inverse_transform(X_test_phi)
    y_test_phi = scaler6.inverse_transform(y_test_phi)

    np.save('data_output/data_xtrain_phi'+ str(j)+'.npy', X_train_phi)
    np.save('data_output/data_ytrain_phi'+ str(j)+'.npy',y_train_phi.reshape(-1,1))
    np.save('data_output/data_xtest_phi'+ str(j)+'.npy',X_test_phi)
    np.save('data_output/data_ytest_phi'+ str(j)+'.npy',y_test_phi.reshape(-1,1))


    #X_train_phi.to_csv('data_output/data_xtrain_phi'+ str(j)+'.csv')
    #y_train_phi.to_csv('data_output/data_ytrain_phi'+ str(j)+'.csv',header = True)
    #X_test_phi.to_csv('data_output/data_xtest_phi'+ str(j)+'.csv')
    #y_test_phi.to_csv('data_output/data_ytest_phi'+ str(j)+'.csv', header = True)
