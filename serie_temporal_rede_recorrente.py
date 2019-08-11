import pandas as pd
base = pd.read_csv('petr4.csv')
base = base.dropna()

base = base.iloc[:, 1].values
import matplotlib.pyplot as plt
plt.plot(base)

periodos = 30
previsao_futura = 1 #horizonte

x = base[0:(len(base) - (len(base) % periodos))]
x_batches = x.reshape(-1, periodos, 1)

y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]
y_batches = y.reshape(-1, periodos, 1)


x_teste = base[-(periodos + previsao_futura):]
x_teste = x_teste[:periodos]
x_teste = x_teste.reshape(-1, periodos, 1)
y_teste = base[-(periodos):]
y_teste = y_teste.reshape(-1, periodos, 1)


import tensorflow as tf
tf.reset_default_graph()

#forecast attribute
entradas = 1
#hidden neurons
neuronios_oculta = 150
#output neurons
neuronios_saida = 1
#placeholders
xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

#celula = tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, activation = tf.nn.relu)
#celula = tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)
#output layer
#celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

def cria_uma_celula():
    return tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

#def cria_varias_celulas():
#   celulas = tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(6)])
#    return tf.contrib.rnn.DropoutWrapper(celulas, output_keep_prob = 0.1)

def cria_varias_celulas():
    return tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(4)])
   

celula = cria_varias_celulas()
#output layer
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)


#output
saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)
#error
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)
#optimizer
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
#training
treinamento = otimizador.minimize(erro)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoca in range(1000):
        _, custo = sess.run([treinamento, erro], feed_dict = {xph: x_batches, yph: y_batches})
        if epoca % 100 == 0:
            print(epoca + 1, 'erro', custo)
            
    previsoes = sess.run(saida_rnn, feed_dict = {xph: x_teste})
    
import numpy as np
y_teste2 = np.ravel(y_teste)
    
previsoes2 = np.ravel(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, previsoes2)


#plt.plot(y_teste2, '*', markersize = 10, label = 'Real Values')
#plt.plot(previsoes2, 'o', label = 'Forecasts')
#plt.legend()


plt.plot(y_teste2, label = 'Real Values')
plt.plot(y_teste2, 'w*', markersize = 10, color = 'red')
plt.plot(previsoes2, label = 'Forecasts')
plt.legend()


    
    