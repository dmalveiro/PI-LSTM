import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Defining GPU device as "device"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Precision
precision = torch.float64

# PREPARE DATA #############################################################

np.random.seed(0)
torch.manual_seed(0)

N = 1000		# number of (x,y,z,t) training examples
H_in = 4		# number of input features
H_out = 1		# number of output features
L = 101			# sequence length (number of time steps)
n_test = 4		# number of test points per space coordinate
N_test_cub = n_test**3	# number of (x,y,z,t) test examples
pi = torch.pi
cos = torch.cos

# Training data (random x,y,z; sequential t)
xyz_train = torch.zeros(N, H_in-1, device=device, dtype=precision)   			# initialize input training set for each time step
xyz_train[:, 0] = torch.rand(N, device=device, dtype=precision) * 2 - 1			# random x-coordinate [0,1] -> [0,2] -> [-1,1]
xyz_train[:, 1] = torch.rand(N, device=device, dtype=precision) * 2 - 1             	# random y-coordinate [0,1] -> [0,2] -> [-1,1]
xyz_train[:, 2] = torch.rand(N, device=device, dtype=precision) * (2*pi)	        # random z-coordinate [0,1] -> [0,2*pi]
xyz_train_rep = xyz_train.unsqueeze(1).repeat(1, L, 1)					# (N,H_in-1) -> (N,1,H_in-1) -> (N,L,H_in-1)
t_train = torch.linspace(0, 1, L, dtype=precision, device=device)			# t-coordinate [0,1] in steps of 0.01
t_train_rep = t_train.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1)			# (L,) -> (1,L) -> (1,L,1) -> (N,L,1)
xyzt_train = torch.cat((xyz_train_rep, t_train_rep), dim=2)				# torch.cat() joins a sequence of tensors along an *existing* dimension (torch.stack() would be along a *new* dimension)
x, y, z, t = xyzt_train[:,:,0], xyzt_train[:,:,1], xyzt_train[:,:,2], xyzt_train[:,:,3]	# Each coordinate corresponds to a (i,j) plan
f_train = (x**3 + 4 * t * x * y**2 + cos(2 * z * t)).unsqueeze(-1)			# output training set: (N,L) -> (N,L,1)

#print("xyz_train", xyz_train.shape)
#print("xyz_train_rep", xyz_train_rep.shape)
#print("t_train", t_train.shape)
#print("t_train_rep", t_train_rep.shape)
#print("xyzt_train", xyzt_train.shape)
#print("f_train", f_train.shape)
#print("xyzt_train device", xyzt_train.device)
#print("f_train device", f_train.device)
#exit()

# Test data (structured grid x,y,z; sequential t)
x, y, z = torch.meshgrid(
          torch.linspace(-1, 1, n_test, dtype=precision, device=device),
          torch.linspace(-1, 1, n_test, dtype=precision, device=device),
          torch.linspace(0, 2*pi, n_test, dtype=precision, device=device),
          indexing='ij')								# Create a N_test_cub mesh, with uniformly spaced mesh points
xyz_test = torch.vstack((torch.ravel(x), torch.ravel(y), torch.ravel(z))).T		# Matrix (N_test_cub,3): each line is the (x,y,z) coordinates of a mesh point
xyz_test_rep = xyz_test.unsqueeze(1).repeat(1, L, 1)                  			# (N_test_cub,H_in-1) -> (N_test_cub,1,H_in-1) -> (N_test_cub,L,H_in-1)
t_test = torch.linspace(0, 1, L, dtype=precision, device=device)                        # t-coordinate [0,1] in steps of 0.01
t_test_rep = t_test.unsqueeze(0).unsqueeze(-1).repeat(N_test_cub, 1, 1)                 # (L,) -> (1,L) -> (1,L,1) -> (N_test_cub,L,1)
xyzt_test = torch.cat((xyz_test_rep, t_test_rep), dim=2)
x, y, z, t = xyzt_test[:,:,0], xyzt_test[:,:,1], xyzt_test[:,:,2], xyzt_test[:,:,3] 	# Each coordinate corresponds to a (i,j) plan
f_test = (x**3 + 4 * t * x * y**2 + cos(2 * z * t)).unsqueeze(-1)                	# output test set: (N_test_cub,L) -> (N_test_cub,L,1)

#print("xyz_test", xyz_test.shape)
#print("xyz_test_rep", xyz_test_rep.shape)
#print("t_test", t_test.shape)
#print("t_test_rep", t_test_rep.shape)
#print("xyzt_test", xyzt_test.shape)
#print("f_test", f_test.shape)
#print("xyzt_test device", xyzt_test.device)
#print("f_test device", f_test.device)
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dtype=precision, device=device)   # forward pass on multilayer LSTM
        self.fc = nn.Linear(hidden_dim, output_dim, dtype=precision, device=device)	# prediction: y=W*h+b

    # Forward pass on the LSTM unit
    # This function runs once per epoch
    def forward(self, x, epoch, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, dtype=precision, device=device)	# initial hidden state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, dtype=precision, device=device)	# initial cell state

        out, (hn, cn) = self.lstm(x, (h0, c0))	# out = hidden-state output at every sequence element
        ######
#        print("x, h0, c0", x.shape, h0.shape, c0.shape)
#        print("out, hn, cn", out.shape, hn.shape, cn.shape)
#        x_save = x.squeeze(-1).numpy()
#        np.savetxt(f'x_ep{epoch}.csv', x_save, delimiter=',')
#        h0_save = h0.squeeze(0).detach().cpu().numpy()
#        c0_save = c0.squeeze(0).detach().cpu().numpy()
#        hn_save = hn.squeeze(0).detach().cpu().numpy()
#        cn_save = cn.squeeze(0).detach().cpu().numpy()
#        np.savetxt(f'h0_ep{epoch}.csv', h0_save, delimiter=',')
#        np.savetxt(f'c0_ep{epoch}.csv', c0_save, delimiter=',')
#        np.savetxt(f'hn_ep{epoch}.csv', hn_save, delimiter=',')
#        np.savetxt(f'cn_ep{epoch}.csv', cn_save, delimiter=',')
        ######
        #out = self.fc(out[:, -1, :])  # call to prediction: out=W*out+b; : -> all N examples of the batch; -1 -> take the last time step; : -> all hidden features
        out = self.fc(out)
        ######
#        print("out_pred", out.shape)
        ######
        return out, hn, cn


# INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER ############################

model = LSTMModel(input_dim=H_in, hidden_dim=64, layer_dim=1, output_dim=H_out)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# TRAIN THE LSTM MODEL #####################################################

num_epochs = 1000
stepoch = 10
num_losses = int(num_epochs / stepoch)
h0, c0 = None, None
loss_history = np.zeros((num_losses,), dtype=np.float64)
loss_i = 0

for epoch in range(num_epochs):

    model.train()		# Set the model in training mode
    optimizer.zero_grad()	# Clear old gradients (from previous epoch)

    # Pass variables to "forward()" function
    # Pass input "trainX", current model's "h0" and "c0"
    # Returns model predictions ("output"), and current model's "hn" and "cn" as the next model's (next epoch) "h0" and "c0"
    outputs, h0, c0 = model(xyzt_train, epoch, h0, c0)

    loss = criterion(outputs, f_train)	# Calculate the loss between the predicted output (training set) and the reference
    loss.backward()			# Gradient of the loss w.r.t. trainable parameters (dL/dθ)
    optimizer.step()			# Updates the parameters: θ(i) = θ(i-1) - η * dL/dθ

    h0, c0 = h0.detach(), c0.detach()	# Take the next model's "h0" and "c0" and disconnect them from the computational graph

    # Print epoch and corresponding loss value
    if (epoch + 1) % stepoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        loss_history[loss_i] = loss.item()
        loss_i += 1

np.savetxt('loss_history.csv', loss_history, delimiter=',')


# EVALUATE PREDICTIONS #####################################################

h0, c0 = None, None

model.eval()	# Set the model in evaluation mode
with torch.no_grad(): f_pred, _, _ = model(xyzt_test, epoch, h0, c0)	# Predict y_pred in the test set (testX)

f_pred = f_pred.squeeze(-1).detach().cpu().numpy()
f_exact = f_test.squeeze(-1).detach().cpu().numpy()
np.savetxt('f_pred.csv', f_pred, delimiter=',')
np.savetxt('f_exact.csv', f_exact, delimiter=',')

l2_error = np.linalg.norm(f_exact - f_pred) / np.linalg.norm(f_exact)
print("L2 error:", l2_error)
