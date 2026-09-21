import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# PREPARE DATA #############################################################

np.random.seed(0)
torch.manual_seed(0)

#t = np.linspace(0, 10000, 100000000)
t = np.linspace(0, 100, 1000)
data = np.sin(t)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)\

seq_length = 10
X, y = create_sequences(data, seq_length)

trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)

train_x = trainX.squeeze(-1).numpy()
train_y = trainY.squeeze(-1).numpy()
np.savetxt('t.csv', t, delimiter=',')
np.savetxt('train_x.csv', train_x, delimiter=',')
np.savetxt('train_y.csv', train_y, delimiter=',')

print("t", t.shape)
print("data", data.shape)
print("trainX", trainX.shape)
print("trainY", trainY.shape)
#exit()


# DEFINE THE LSTM MODEL ####################################################

# input_dim = number of input features (equivalent to the number of nodes at the input layer of an MLP)
# hidden_dim = number of features in the hidden state
# layer_dim = number of stacked LSTMs: LSTM_1 -> LSTM_2 -> ... -> LSTM_layer_dim (equivalent to the number of hidden layers in an MLP)
# Each LSTM_i repeatedly/sequentially processes each time step / sequence element at a time
# output_dim = number of outputs from the "Linear" layer (output_dim = number of outputs??)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)   # multilayer LSTM (last hidden state: h)
        self.fc = nn.Linear(hidden_dim, output_dim)				  # prediction: y=W*h+b

    # Forward pass on the LSTM unit
    # This function runs once per epoch
    def forward(self, x, epoch, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)	# initial hidden state
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)	# initial cell state

        out, (hn, cn) = self.lstm(x, (h0, c0))	# out = hidden-state output at every sequence element
        ######
#        print("x, h0, c0", x.shape, h0.shape, c0.shape)
#        print("out, hn, cn", out.shape, hn.shape, cn.shape)
#        x_save = x.squeeze(-1).numpy()
#        np.savetxt(f'x_ep{epoch}.csv', x_save, delimiter=',')
#        h0_save = h0.squeeze(0).detach().numpy()
#        c0_save = c0.squeeze(0).detach().numpy()
#        hn_save = hn.squeeze(0).detach().numpy()
#        cn_save = cn.squeeze(0).detach().numpy()
#        np.savetxt(f'h0_ep{epoch}.csv', h0_save, delimiter=',')
#        np.savetxt(f'c0_ep{epoch}.csv', c0_save, delimiter=',')
#        np.savetxt(f'hn_ep{epoch}.csv', hn_save, delimiter=',')
#        np.savetxt(f'cn_ep{epoch}.csv', cn_save, delimiter=',')
        ######
        out = self.fc(out[:, -1, :])  # call to prediction: out=W*out+b; : -> all N examples of the batch; -1 -> take the last time step; : -> all hidden features
        ######
#        print("out_pred", out.shape)
        ######
        return out, hn, cn


# INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER ############################

model = LSTMModel(input_dim=1, hidden_dim=64, layer_dim=1, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# TRAIN THE LSTM MODEL #####################################################

num_epochs = 100
h0, c0 = None, None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, epoch, h0, c0)

    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    h0, c0 = h0.detach(), c0.detach()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')


# EVALUATE PREDICTIONS #####################################################


