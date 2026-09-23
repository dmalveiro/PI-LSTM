import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# PREPARE DATA #############################################################

np.random.seed(0)
torch.manual_seed(0)

#t = np.linspace(0, 10000, 100000000)
t = np.linspace(0, 100, 1000)
t1 = np.linspace(330, 530, 1000)
data = np.sin(t)
data1 = np.sin(t1)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)\

def create_sequences1(data1, seq_length):
    xs1, ys1 = [], []
    for i in range(len(data1) - seq_length):
        x1 = data1[i:(i + seq_length)]
        y1 = data1[i + seq_length]
        xs1.append(x1)
        ys1.append(y1)
    return np.array(xs1), np.array(ys1)\

seq_length = 10
X, y = create_sequences(data, seq_length)
X1, y1 = create_sequences1(data1, seq_length)

trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)

testX = torch.tensor(X1[:, :, None], dtype=torch.float32)
testY = torch.tensor(y1[:, None], dtype=torch.float32)

train_x = trainX.squeeze(-1).numpy()
train_y = trainY.squeeze(-1).numpy()
np.savetxt('t.csv', t, delimiter=',')
np.savetxt('train_x.csv', train_x, delimiter=',')
np.savetxt('train_y.csv', train_y, delimiter=',')

#print("t", t.shape)
#print("data", data.shape)
#print("trainX", trainX.shape)
#print("trainY", trainY.shape)
#exit()


# DEFINE THE LSTM MODEL ####################################################

# input_dim = number of input features (analogous to the number of nodes at the input layer of a discrete-time (DT) MLP)
# hidden_dim = number of features in the hidden state (analogous to the number of nodes per hidden layer of a DT-MLP)
# layer_dim = number of stacked LSTMs: LSTM_1 -> LSTM_2 -> ... -> LSTM_layer_dim (analogous to the number of hidden layers in DT-MLP)
# Each LSTM_i repeatedly/sequentially processes each time step / sequence element at a time
# output_dim = number of outputs from the "Linear" layer (analogous to the number of nodes at the output layer of a DT-MLP)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)   # forward pass on multilayer LSTM network (last hidden state: h)
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

num_epochs = 1000
stepoch = 10
num_losses = int(num_epochs / stepoch)
h0, c0 = None, None
loss_history = np.zeros((num_losses,))
loss_i = 0

for epoch in range(num_epochs):

    model.train()		# Set the model in training mode
    optimizer.zero_grad()	# Clear old gradients (from previous epoch)

    # Pass variables to "forward()" function
    # Pass input "trainX", current model's "h0" and "c0"
    # Returns model predictions ("output"), and current model's "hn" and "cn" as the next model's (next epoch) "h0" and "c0"
    outputs, h0, c0 = model(trainX, epoch, h0, c0)

    loss = criterion(outputs, trainY)	# Calculate the loss between the predicted output (training set) and the reference
    loss.backward()			# Gradient of the loss w.r.t. trainable parameters (dL/dθ)
    optimizer.step()			# Updates the parameters: θ(i) = θ(i-1) - η * dL/dθ

    h0, c0 = h0.detach(), c0.detach()	# Take the next model's "h0" and "c0" and disconnect them from the computational graph

    # Print epoch and corresponding loss value
    if (epoch + 1) % stepoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
        loss_history[loss_i] = loss.item()
        loss_i += 1

np.savetxt('loss_history.csv', loss_history.8f, delimiter=',')

# EVALUATE PREDICTIONS #####################################################

model.eval()	# Set the model in evaluation mode
with torch.no_grad(): y_pred, _, _ = model(testX, epoch, h0, c0)	# Predict y_pred in the test set (testX)

y_pred = y_pred.squeeze(-1).detach().numpy()
y_exact = testY.squeeze(-1).detach().numpy()
np.savetxt('y_pred.csv', y_pred, delimiter=',')
np.savetxt('y_exact.csv', y_exact, delimiter=',')

l2_error = np.linalg.norm(y_exact - y_pred) / np.linalg.norm(y_exact)
print("L2 error", l2_error)
