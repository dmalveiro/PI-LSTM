import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

num_layers = 2
hidden_size = 7
L = 11
num_epochs = 3

xyzt = np.stack((x, y, z, t), dim=2)    # x_t

for j in range(L):
    for k in range(num_layers):

        # LSTM weights
        W_ih, W_hh = W_ih[k], W_hh[k]    # from joined l0 and l1 at a dictionary
        W_ii = W_ih[0:hidden_size, :]
        W_if = W_ih[hidden_size:2*hidden_size, :]
        W_ig = W_ih[2*hidden_size:3*hidden_size, :]
        W_ig = W_ih[3*hidden_size:4*hidden_size, :]
        W_hi = W_hh[0:hidden_size, :]
        W_hf = W_hh[hidden_size:2*hidden_size, :]
        W_hg = W_hh[2*hidden_size:3*hidden_size, :]
        W_hg = W_hh[3*hidden_size:4*hidden_size, :]

        # LSTM biases
        b_ih, b_hh = b_ih[k], b_hh[k]
        b_ii = b_ih[0:hidden_size]
        b_if = b_ih[hidden_size:2*hidden_size]
        b_ig = b_ih[2*hidden_size:3*hidden_size]
        b_ig = b_ih[3*hidden_size:4*hidden_size]
        b_hi = b_hh[0:hidden_size]
        b_hf = b_hh[hidden_size:2*hidden_size]
        b_hg = b_hh[2*hidden_size:3*hidden_size]
        b_hg = b_hh[3*hidden_size:4*hidden_size]

        if k > 0:
            x_t = out ###### CHANGE THIS
        else:
            x_t = xyzt

        i_t[j,k] = sigmoid(W_ii @ np.transpose(x_t[:, j, :]) + b_ii + W_hi @ np.transpose(h_t1[:, j, :]) + b_hi)
        f_t[j,k] = sigmoid(W_if @ np.transpose(x_t[:, j, :]) + b_if + W_hf @ np.transpose(h_t1[:, j, :]) + b_hf)
        g_t[j,k] = tanh(W_ig @ np.transpose(x_t[:, j, :]) + b_ig + W_hg @ np.transpose(h_t1[:, j, :]) + b_hg)
        o_t[j,k] = sigmoid(W_io @ np.transpose(x_t[:, j, :]) + b_io + W_ho @ np.transpose(h_t1[:, j, :]) + b_ho)
        c_t = f_t * c_t1 + i_t * g_t
        h_t = o_t * tanh(c_t)
