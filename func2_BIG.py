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

#N = 5
N = 5000		# number of (x,y,z,t) training examples
H_in = 4		# number of input features
H_out = 1		# number of output features
L = 11			# sequence length (number of time steps)
n_test = 8		# number of test points per space coordinate
N_test_cub = n_test**3	# number of (x,y,z,t) test examples
pi = torch.pi
cos = torch.cos
t_i, t_f = 0, 1
dt = (t_f-t_i)/(L-1)

# Training data (random x,y,z; sequential t)
xyz_train = torch.zeros(N, H_in-1, device=device, dtype=precision)   			# initialize input training set for each time step
xyz_train[:, 0] = torch.rand(N, device=device, dtype=precision) * 2 - 1			# random x-coordinate [0,1] -> [0,2] -> [-1,1]
xyz_train[:, 1] = torch.rand(N, device=device, dtype=precision) * 2 - 1             	# random y-coordinate [0,1] -> [0,2] -> [-1,1]
xyz_train[:, 2] = torch.rand(N, device=device, dtype=precision) * (2*pi)	        # random z-coordinate [0,1] -> [0,2*pi]
xyz_train_rep = xyz_train.unsqueeze(1).repeat(1, L, 1)					# (N,H_in-1) -> (N,1,H_in-1) -> (N,L,H_in-1)
t_train = torch.linspace(t_i, t_f, L, dtype=precision, device=device)			# t-coordinate [0,1] in steps of 0.01
t_train_rep = t_train.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1)			# (L,) -> (1,L) -> (1,L,1) -> (N,L,1)
xyzt_train = torch.cat((xyz_train_rep, t_train_rep), dim=2)				# torch.cat() joins a sequence of tensors along an *existing* dimension (torch.stack() would be along a *new* dimension)
x, y, z, t = xyzt_train[:,:,0], xyzt_train[:,:,1], xyzt_train[:,:,2], xyzt_train[:,:,3]	# Each coordinate corresponds to a (i,j) plan
f_train_exact = (x**3 + 4 * t * x * y**2 + cos(2 * z * t)).unsqueeze(-1)			# output training set: (N,L) -> (N,L,1)
np.savetxt("xyz_train.csv", xyz_train.detach().cpu().numpy(), delimiter=',')

# Test data (structured grid x,y,z; sequential t)
x, y, z = torch.meshgrid(
          torch.linspace(-1, 1, n_test, dtype=precision, device=device),
          torch.linspace(-1, 1, n_test, dtype=precision, device=device),
          torch.linspace(0, 2*pi, n_test, dtype=precision, device=device),
          indexing='ij')								# Create a N_test_cub mesh, with uniformly spaced mesh points
xyz_test = torch.vstack((torch.ravel(x), torch.ravel(y), torch.ravel(z))).T		# Matrix (N_test_cub,3): each line is the (x,y,z) coordinates of a mesh point
xyz_test_rep = xyz_test.unsqueeze(1).repeat(1, L, 1)                  			# (N_test_cub,H_in-1) -> (N_test_cub,1,H_in-1) -> (N_test_cub,L,H_in-1)
t_test = torch.linspace(t_i, t_f, L, dtype=precision, device=device)                        # t-coordinate [0,1] in steps of 0.01
t_test_rep = t_test.unsqueeze(0).unsqueeze(-1).repeat(N_test_cub, 1, 1)                 # (L,) -> (1,L) -> (1,L,1) -> (N_test_cub,L,1)
xyzt_test = torch.cat((xyz_test_rep, t_test_rep), dim=2)
x, y, z, t = xyzt_test[:,:,0], xyzt_test[:,:,1], xyzt_test[:,:,2], xyzt_test[:,:,3] 	# Each coordinate corresponds to a (i,j) plan
f_test_exact = (x**3 + 4 * t * x * y**2 + cos(2 * z * t)).unsqueeze(-1)                	# output test set: (N_test_cub,L) -> (N_test_cub,L,1)
np.savetxt("xyz_test.csv", xyz_test.detach().cpu().numpy(), delimiter=',')


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
        out = self.fc(out)	# call to prediction: for all time steps, y=W*h+b, where W and b are the parameters of the linear layer

        return out, hn, cn


# COMPUTE DERIVATIVES ######################################################

# not needed for this model; this is just to see if the derivatives of f are well calculated

def compute_gradients(f, xyzt, epoch, L, dt):

    def derivative(aa, bb, create=True):
        return torch.autograd.grad(aa, bb, grad_outputs=torch.ones_like(aa), retain_graph=True, create_graph=create)[0]

    os.makedirs("GRADIENTS", exist_ok=True)    # create folder "GRADIENTS" in current location

    # Lists to accumulate the result for each time step
    fx, fy, fz = [], [], []
    fxx, fyy, fzz = [], [], []
    fx_exact, fy_exact, fz_exact = [], [], []
    fxx_exact, fyy_exact, fzz_exact = [], [], []
    gradients = []

    names = ["fx", "fy", "fz", "fxx", "fyy", "fzz", "fx_exact", "fy_exact", "fz_exact", "fxx_exact", "fyy_exact", "fzz_exact"]
    data = [fx, fy, fz, fxx, fyy, fzz, fx_exact, fy_exact, fz_exact, fxx_exact, fyy_exact, fzz_exact]

    for t_i in range(L):
#    for t_i in range(10, 11):

        print("t_i", t_i)

        t = t_i * dt

        # Only one f component
        f_t = f[:, t_i, 0:1]

#        f = f.detach().cpu().numpy()
#        print("f", f.shape)
#        print("xyzt", xyzt.shape)
#        exit()

        ######### TEST ##################

#        xx, yy, zz, tt = xyzt[:, t_i, 0:1], xyzt[:, t_i, 1:2], xyzt[:, t_i, 2:3], xyzt[:, t_i, 3:4]
#        f_t = xx**3 + 4 * tt * xx * yy**2 + cos(2 * zz * tt)

        #################################

        # Space derivatives
#        fx_t = derivative(f_t, xyzt, create=True)[:, t_i, 0:1]
#        fy_t = derivative(f_t, xyzt, create=True)[:, t_i, 1:2]
#        fz_t = derivative(f_t, xyzt, create=True)[:, t_i, 2:3]
#        fxx_t = derivative(fx_t, xyzt, create=False)[:, t_i, 0:1]
#        fyy_t = derivative(fy_t, xyzt, create=False)[:, t_i, 1:2]
#        fzz_t = derivative(fz_t, xyzt, create=False)[:, t_i, 2:3]

        #########
        fx_all = derivative(f_t, xyzt, create=True)[:, :t_i+1, 0:1]
        fy_all = derivative(f_t, xyzt, create=True)[:, :t_i+1, 1:2]
        fz_all = derivative(f_t, xyzt, create=True)[:, :t_i+1, 2:3]
        fxx_all = derivative(fx_all, xyzt, create=False)[:, :t_i+1, 0:1]
        fyy_all = derivative(fy_all, xyzt, create=False)[:, :t_i+1, 1:2]
        fzz_all = derivative(fz_all, xyzt, create=False)[:, :t_i+1, 2:3]

        fx_t = torch.sum(fx_all, 1)
        fy_t = torch.sum(fy_all, 1)
        fz_t = torch.sum(fz_all, 1)
        fxx_t = torch.sum(fxx_all, 1)
        fyy_t = torch.sum(fyy_all, 1)
        fzz_t = torch.sum(fzz_all, 1)
        #########

        fx_t = fx_t.detach().cpu().numpy().squeeze(-1)	# since the output is only f (not f=(f_1,f_2,...,f_n)), the format of f is [N, L, 1], and thus, eliminate the last dimension
        fy_t = fy_t.detach().cpu().numpy().squeeze(-1)
        fz_t = fz_t.detach().cpu().numpy().squeeze(-1)
        fxx_t = fxx_t.detach().cpu().numpy().squeeze(-1)
        fyy_t = fyy_t.detach().cpu().numpy().squeeze(-1)
        fzz_t = fzz_t.detach().cpu().numpy().squeeze(-1)

        #########
#        fx_all = fx_all.detach().cpu().numpy().squeeze(-1)
#        fy_all = fy_all.detach().cpu().numpy().squeeze(-1)
#        fz_all = fz_all.detach().cpu().numpy().squeeze(-1)
#        fxx_all = fxx_all.detach().cpu().numpy().squeeze(-1)
#        fyy_all = fyy_all.detach().cpu().numpy().squeeze(-1)
#        fzz_all = fzz_all.detach().cpu().numpy().squeeze(-1)
        #########

        # Exact derivatives for reference
        x_t = xyzt.detach().cpu().numpy()[:, t_i, 0:1]
        y_t = xyzt.detach().cpu().numpy()[:, t_i, 1:2]
        z_t = xyzt.detach().cpu().numpy()[:, t_i, 2:3]
        fx_t_exact = 3*x_t**2 + 4*t*y_t**2
        fy_t_exact = 8*t*x_t*y_t
        fz_t_exact = -2*t*np.sin(2*z_t*t)
        fxx_t_exact = 6*x_t
        fyy_t_exact = 8*t*x_t
        fzz_t_exact = -4*t**2*np.cos(2*z_t*t)

        fx_t_exact = fx_t_exact.squeeze(-1)
        fy_t_exact = fy_t_exact.squeeze(-1)
        fz_t_exact = fz_t_exact.squeeze(-1)
        fxx_t_exact = fxx_t_exact.squeeze(-1)
        fyy_t_exact = fyy_t_exact.squeeze(-1)
        fzz_t_exact = fzz_t_exact.squeeze(-1)

#        print("z_t", z_t.shape)

        data_t = [fx_t, fy_t, fz_t, fxx_t, fyy_t, fzz_t, fx_t_exact, fy_t_exact, fz_t_exact, fxx_t_exact, fyy_t_exact, fzz_t_exact]

        for j, k in zip(data, data_t):
            j.append(k)

#        fx.append(fx_t)
#        fy.append(fy_t)
#        fz.append(fz_t)
#        fxx.append(fxx_t)
#        fyy.append(fyy_t)
#        fzz.append(fzz_t)
#        fx_exact.append(fx_t_exact)
#        fy_exact.append(fy_t_exact)
#        fz_exact.append(fz_t_exact)
#        fxx_exact.append(fxx_t_exact)
#        fyy_exact.append(fyy_t_exact)
#        fzz_exact.append(fzz_t_exact)

        #########
#        np.savetxt(f'GRADIENTS/fx_all_ep{epoch}.csv', fx_all, delimiter=',')
#        np.savetxt(f'GRADIENTS/fy_all_ep{epoch}.csv', fy_all, delimiter=',')
#        np.savetxt(f'GRADIENTS/fz_all_ep{epoch}.csv', fz_all, delimiter=',')
#        np.savetxt(f'GRADIENTS/fxx_all_ep{epoch}.csv', fxx_all, delimiter=',')
#        np.savetxt(f'GRADIENTS/fyy_all_ep{epoch}.csv', fyy_all, delimiter=',')
#        np.savetxt(f'GRADIENTS/fzz_all_ep{epoch}.csv', fzz_all, delimiter=',')
        #########

#    for i, j in zip(names, data):
    for n, (i, j) in enumerate(zip(names, data)):
        j = np.stack(j, axis=1)
        if n >= 6:
            deriv_exact = j
#            print("deriv_exact", deriv_exact.shape)
            deriv_autograd = np.stack(data[n-6], axis=1)
            l2_error = np.linalg.norm(deriv_exact - deriv_autograd) / np.linalg.norm(deriv_exact)
            print(f"L2 error between {i} and {names[n-6]}_autograd:", l2_error)
        np.savetxt(f'GRADIENTS/{i}_ep{epoch}.csv', j, delimiter=',')
        gradients.append(j)

#    fx = np.stack(fx, axis=1)
#    fy = np.stack(fy, axis=1)
#    fz = np.stack(fz, axis=1)
#    fxx = np.stack(fxx, axis=1)
#    fyy = np.stack(fyy, axis=1)
#    fzz = np.stack(fzz, axis=1)
#    fx_exact = np.stack(fx_exact, axis=1)
#    fy_exact = np.stack(fy_exact, axis=1)
#    fz_exact = np.stack(fz_exact, axis=1)
#    fxx_exact = np.stack(fxx_exact, axis=1)
#    fyy_exact = np.stack(fyy_exact, axis=1)
#    fzz_exact = np.stack(fzz_exact, axis=1)
#    np.savetxt(f'GRADIENTS/fx_ep{epoch}.csv', fx, delimiter=',')
#    np.savetxt(f'GRADIENTS/fy_ep{epoch}.csv', fy, delimiter=',')
#    np.savetxt(f'GRADIENTS/fz_ep{epoch}.csv', fz, delimiter=',')
#    np.savetxt(f'GRADIENTS/fxx_ep{epoch}.csv', fxx, delimiter=',')
#    np.savetxt(f'GRADIENTS/fyy_ep{epoch}.csv', fyy, delimiter=',')
#    np.savetxt(f'GRADIENTS/fzz_ep{epoch}.csv', fzz, delimiter=',')
#    np.savetxt(f'GRADIENTS/fx_exact_ep{epoch}.csv', fx_exact, delimiter=',')
#    np.savetxt(f'GRADIENTS/fy_exact_ep{epoch}.csv', fy_exact, delimiter=',')
#    np.savetxt(f'GRADIENTS/fz_exact_ep{epoch}.csv', fz_exact, delimiter=',')
#    np.savetxt(f'GRADIENTS/fxx_exact_ep{epoch}.csv', fxx_exact, delimiter=',')
#    np.savetxt(f'GRADIENTS/fyy_exact_ep{epoch}.csv', fyy_exact, delimiter=',')
#    np.savetxt(f'GRADIENTS/fzz_exact_ep{epoch}.csv', fzz_exact, delimiter=',')

    return gradients


# INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER ############################

#model = LSTMModel(input_dim=H_in, hidden_dim=80, layer_dim=1, output_dim=H_out)
model = LSTMModel(input_dim=H_in, hidden_dim=200, layer_dim=1, output_dim=H_out)
criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# TRAIN THE LSTM MODEL #####################################################

#num_epochs = 1850
#num_epochs = 49536
num_epochs = 36016
stepoch = 1
num_losses = int(num_epochs / stepoch)
loss_history = np.zeros((num_losses,), dtype=np.float64)
loss_i = 0

for epoch in range(num_epochs):

    h0, c0 = None, None

    model.train()		# Set the model in training mode
    optimizer.zero_grad()	# Clear old gradients (from previous epoch)

    # 10
    # this is needed for the gradients calculation (comment if not computing gradients)
    xyzt_train = xyzt_train.detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False): f_train_pred, h0, c0 = model(xyzt_train, epoch, h0, c0)
    #######

    # Pass variables to "forward()" function
    # Pass input "trainX", current model's "h0" and "c0"
    # Returns model predictions (f, "output"), and current model's "hn" and "cn" as the next model's (next epoch) "h0" and "c0"
#    f_train_pred, h0, c0 = model(xyzt_train, epoch, h0, c0)

    # 8
    # compute gradients (not needed)
#    print("f_train_pred", f_train_pred.shape, f_train_pred.device, f_train_pred.requires_grad, f_train_pred.is_leaf, f_train_pred.grad)
#    print("XYZT_TRAIN", xyzt_train.shape)
#    if epoch == num_epochs: gradients = compute_gradients(f_train_pred, xyzt_train, epoch, L, dt)
    #######

    loss = criterion(f_train_pred, f_train_exact)	# Calculate the loss between the predicted output (training set) and the reference
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

epoch = 'TEST'

model.eval()    # Set the model in evaluation mode

#
xyzt_test = xyzt_train
f_test_exact = f_train_exact

# 11
# this is needed for the gradients calculation (comment if not computing gradients)
xyzt_test = xyzt_test.detach().requires_grad_(True)
with torch.backends.cudnn.flags(enabled=False): f_test_pred, _, _ = model(xyzt_test, epoch, h0, c0)	# cuDNN is not compatible with double backward pass (2nd derivatives) in LSTMs
#######

#with torch.no_grad(): f_test_pred, _, _ = model(xyzt_test, epoch, h0, c0)	# Predict y_pred in the test set. No need for gradients.

# 9
# compute gradients (not needed)
#print("f_test_pred", f_test_pred.shape, f_test_pred.device, f_test_pred.requires_grad, f_test_pred.is_leaf, f_test_pred.grad)
#print("XYZT_TEST", xyzt_test.shape)
gradients = compute_gradients(f_test_pred, xyzt_test, epoch, L, dt)
#######

f_test_pred = f_test_pred.squeeze(-1).detach().cpu().numpy()
f_test_exact = f_test_exact.squeeze(-1).detach().cpu().numpy()

l2_error = np.linalg.norm(f_test_exact - f_test_pred) / np.linalg.norm(f_test_exact)
print("L2 error:", l2_error)

np.savetxt('f_test_pred.csv', f_test_pred, delimiter=',')
np.savetxt('f_test_exact.csv', f_test_exact, delimiter=',')


# DERIVATIVES THROUGH FINITE DIFFERENCING ##################################

epsilon = 1e-5              # very small number

names = ["fx", "fy", "fz", "fxx", "fyy", "fzz"]

xyzt = xyzt_train

for i in range(3):

    j = i + 3

    xyzt_plus = xyzt.clone()
    xyzt_minus = xyzt.clone()

    xyzt_plus[:, :, i] += epsilon
    xyzt_minus[:, :, i] -= epsilon

    #np.savetxt('xyzt_train.csv', xyzt_train[:, :, 0].detach().cpu().numpy(), delimiter=',')
    #np.savetxt('xyzt_plus.csv', xyzt_plus[:, :, 0].detach().cpu().numpy(), delimiter=',')
    #np.savetxt('xyzt_minus.csv', xyzt_minus[:, :, 0].detach().cpu().numpy(), delimiter=',')

    with torch.no_grad():
        f_plus, _, _ = model(xyzt_plus, epoch, None, None)
        f_minus, _, _ = model(xyzt_minus, epoch, None, None)
        f, _, _ = model(xyzt, epoch, None, None)

    # 1st derivatives: central differences
    fd1 = (f_plus[:, :, 0] - f_minus[:, :, 0]) / (2 * epsilon)
    fd1_ = fd1.detach().cpu().numpy()

    # 2nd derivatives: central differences
    fd2 = (f_plus[:, :, 0] - 2 * f[:, :, 0] + f_minus[:, :, 0]) / (epsilon**2)
    fd2_ = fd2.detach().cpu().numpy()

    np.savetxt(f'{names[i]}_fd.csv', fd1_, delimiter=',')
    np.savetxt(f'{names[j]}_fd.csv', fd2_, delimiter=',')

    deriv1_autograd = gradients[i]
    deriv2_autograd = gradients[j]
    l2_error_1 = np.linalg.norm(fd1_ - deriv1_autograd) / np.linalg.norm(fd1_)
    l2_error_2 = np.linalg.norm(fd2_ - deriv2_autograd) / np.linalg.norm(fd2_)

    print(f"L2 error between {names[i]}_fd and {names[i]}_autograd:", l2_error_1)
    print(f"L2 error between {names[j]}_fd and {names[j]}_autograd:", l2_error_2)
