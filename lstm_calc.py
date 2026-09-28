import numpy as np
import pandas as pd

L = 11
N = 5
H_in = 4
H_out = 1
num_layers = 2
hidden_size = 7
epoch = 1	# desired epoch

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Weights and biases - linear layer
W_fc = f"RUN2/W_fc_ep{epoch}.csv"
W_fc = pd.read_csv(W_fc, header=None).to_numpy()
b_fc = f"RUN2/b_fc_ep{epoch}.csv"
b_fc = pd.read_csv(b_fc, header=None).to_numpy()

h_out = []

for j in range(num_layers):

    xt, out, h_j = [], [], []
    h_0 = np.zeros((hidden_size, N))
    c_0 = np.zeros((hidden_size, N))

    for v in ["x", "y", "z", "t"]:
        xt_i = pd.read_csv(f"RUN2/{v}_ep{epoch}.csv", header=None).to_numpy()
        xt.append(xt_i)

    for hd in range(hidden_size):
        out_hd = pd.read_csv(f"RUN2/out_l{j}_hid{hd}_ep{epoch}.csv", header=None).to_numpy()
        out.append(out_hd)

    W_ih = pd.read_csv(f"RUN2/W_ih_l{j}_ep{epoch}.csv", header=None).to_numpy()
    W_hh = pd.read_csv(f"RUN2/W_hh_l{j}_ep{epoch}.csv", header=None).to_numpy()
    b_ih = pd.read_csv(f"RUN2/b_ih_l{j}_ep{epoch}.csv", header=None).to_numpy()
    b_hh = pd.read_csv(f"RUN2/b_hh_l{j}_ep{epoch}.csv", header=None).to_numpy()

    xt = np.stack(xt, axis=2)
    out = np.stack(out, axis=2)

    # LSTM weights
    W_ii = W_ih[0:hidden_size, :]
    W_if = W_ih[hidden_size:2*hidden_size, :]
    W_ig = W_ih[2*hidden_size:3*hidden_size, :]
    W_io = W_ih[3*hidden_size:4*hidden_size, :]
    W_hi = W_hh[0:hidden_size, :]
    W_hf = W_hh[hidden_size:2*hidden_size, :]
    W_hg = W_hh[2*hidden_size:3*hidden_size, :]
    W_ho = W_hh[3*hidden_size:4*hidden_size, :]

    # LSTM biases
    b_ii = b_ih[0:hidden_size]
    b_if = b_ih[hidden_size:2*hidden_size]
    b_ig = b_ih[2*hidden_size:3*hidden_size]
    b_io = b_ih[3*hidden_size:4*hidden_size]
    b_hi = b_hh[0:hidden_size]
    b_hf = b_hh[hidden_size:2*hidden_size]
    b_hg = b_hh[2*hidden_size:3*hidden_size]
    b_ho = b_hh[3*hidden_size:4*hidden_size]

#    print(W_ii.shape, b_ii.shape)

    #if j == 0:
    #    x_t_ = xt	# (x,y,z,t) input
    #else:
    #    x_t_ = h		# LSTM hidden output from the previous layer

#    print(x_t.shape)
#    print(x_t[:,3,:].shape)
#    print(h_0.shape, c_0.shape)
#    exit()

    for t in range(L):
        print("t =", t, "-----------------")
        if t == 0:
            h_t = h_0
            c_t = c_0
        if j == 0:
            x_t = np.transpose(xt[:, t, :])
        else:
            x_t = h[:, t, :]
        print("h_t", h_t.shape)
        print(x_t.shape)
        i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_t + b_hi)    # h_t is still from t-1 (only updated at the end of each iteration)
        f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_t + b_hf)
        g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_t + b_hg)
        o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_t + b_ho)
        print("f_t, c_t, i_t, g_t, o_t", f_t.shape, c_t.shape, i_t.shape, g_t.shape, o_t.shape)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * tanh(c_t)
        #h_j.append(np.transpose(h_t))
        h_j.append(h_t)

    h = np.stack(h_j, axis=1)   # h from current layer j for all time steps

#    h_out.append(h)
#h = np.concatenate(h_out, axis=0)	# equivalent of torch.cat

print("h, W_fc, b_fc", h.shape, W_fc.shape, b_fc.shape)

y_out = []

for t in range(L):
   out = h[:, t, :]
   print("out", out.shape)
   y = W_fc @ out + b_fc
   y_out.append(y)

y = np.stack(y_out, axis=1)

print("y", y.shape)

y = np.transpose(y.squeeze(0))
np.savetxt('y.csv', y, delimiter=',')

print("END")
