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

N = 5		# number of (x,y,z,t) training examples
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

# 1
#print("xyz_train", xyz_train.shape)
#print("xyz_train_rep", xyz_train_rep.shape)
#print("t_train", t_train.shape)
#print("t_train_rep", t_train_rep.shape)
#print("xyzt_train", xyzt_train.shape, xyzt_train.device, xyzt_train.requires_grad, xyzt_train.is_leaf, xyzt_train.grad)
#print("f_train_exact", f_train_exact.shape, f_train_exact.device, f_train_exact.requires_grad, f_train_exact.is_leaf, f_train_exact.grad)
#exit()

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

# 2
#print("xyz_test", xyz_test.shape)
#print("xyz_test_rep", xyz_test_rep.shape)
#print("t_test", t_test.shape)
#print("t_test_rep", t_test_rep.shape)
#print("xyzt_test", xyzt_test.shape, xyzt_test.device, xyzt_test.requires_grad, xyzt_test.is_leaf, xyzt_test.grad)
#print("f_test_exact", f_test_exact.shape, f_test_exact.device, f_test_exact.requires_grad, f_test_exact.is_leaf, f_test_exact.grad)
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

        # 4
        #print("EPOCH", epoch, "---------------")
        #print("x, h0, c0", x.shape, h0.shape, c0.shape)
        #print("x", x.device, x.requires_grad, x.is_leaf, x.grad)
        #print("h0", h0.device, h0.requires_grad, h0.is_leaf, h0.grad)
        #print("c0", c0.device, c0.requires_grad, c0.is_leaf, c0.grad)
        ######

        out, (hn, cn) = self.lstm(x, (h0, c0))	# out = hidden-state output at every sequence element

        # 5
        # for every epoch, it saves the parameters before the parameter update by the Adam optimizer
#        print("EPOCH", epoch, "---------------")
#        print("xyzt, h0, c0", x.shape, h0.shape, c0.shape)
#        print("out, hn, cn", out.shape, hn.shape, cn.shape)
#        for k in range(self.layer_dim):
#            W_ih = getattr(model.lstm, f"weight_ih_l{k}")
#            W_hh = getattr(model.lstm, f"weight_hh_l{k}")
#            b_ih = getattr(model.lstm, f"bias_ih_l{k}")
#            b_hh = getattr(model.lstm, f"bias_hh_l{k}")
#            print(f"W_ih_l{k}:", W_ih.shape)
#            print(f"W_hh_l{k}:", W_hh.shape)
#            print(f"b_ih_l{k}:", b_ih.shape)
#            print(f"b_hh_l{k}:", b_hh.shape)
#            W_ih_ = W_ih.detach().cpu().numpy()
#            W_hh_ = W_hh.detach().cpu().numpy()
#            b_ih_ = b_ih.detach().cpu().numpy()
#            b_hh_ = b_hh.detach().cpu().numpy()
#            np.savetxt(f'W_ih_l{k}_ep{epoch}.csv', W_ih_, delimiter=',')
#            np.savetxt(f'W_hh_l{k}_ep{epoch}.csv', W_hh_, delimiter=',')
#            np.savetxt(f'b_ih_l{k}_ep{epoch}.csv', b_ih_, delimiter=',')
#            np.savetxt(f'b_hh_l{k}_ep{epoch}.csv', b_hh_, delimiter=',')
#        xyzt_ = x.detach().cpu().numpy()
#        h0_ = h0.detach().cpu().numpy()
#        c0_ = c0.detach().cpu().numpy()
#        out_ = out.detach().cpu().numpy()
#        hn_ = hn.detach().cpu().numpy()
#        cn_ = cn.detach().cpu().numpy()
#        np.savetxt(f'x_ep{epoch}.csv', xyzt_[:, :, 0], delimiter=',')
#        np.savetxt(f'y_ep{epoch}.csv', xyzt_[:, :, 1], delimiter=',')
#        np.savetxt(f'z_ep{epoch}.csv', xyzt_[:, :, 2], delimiter=',')
#        np.savetxt(f't_ep{epoch}.csv', xyzt_[:, :, 3], delimiter=',')
#        for hd in range(self.hidden_dim):
#            np.savetxt(f'h0_hid{hd}_ep{epoch}.csv', h0_[:, :, hd], delimiter=',')
#            np.savetxt(f'c0_hid{hd}_ep{epoch}.csv', c0_[:, :, hd], delimiter=',')
#            np.savetxt(f'hn_hid{hd}_ep{epoch}.csv', hn_[:, :, hd], delimiter=',')
#            np.savetxt(f'cn_hid{hd}_ep{epoch}.csv', cn_[:, :, hd], delimiter=',')
#            np.savetxt(f'out_hid{hd}_ep{epoch}.csv', out_[:, :, hd], delimiter=',')
        ######

        # y = self.fc(h)
        out = self.fc(out)	# call to prediction: for all time steps, y=W*h+b, where W and b are the parameters of the linear layer

        # 6
#        print("f:", out.shape)
#        f_ = out.squeeze(-1).detach().cpu().numpy()
#        np.savetxt(f'f_ep{epoch}.csv', f_, delimiter=',')
#        W_fc = model.fc.weight
#        b_fc = model.fc.bias
#        print(f"W_fc:", W_fc.shape)
#        print(f"b_fc:", b_fc.shape)
#        W_fc_ = W_fc.detach().cpu().numpy()
#        b_fc_ = b_fc.detach().cpu().numpy()
#        np.savetxt(f'W_fc_ep{epoch}.csv', W_fc_, delimiter=',')
#        np.savetxt(f'b_fc_ep{epoch}.csv', b_fc_, delimiter=',')
        ######

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

    names = ["fx", "fy", "fz", "fxx", "fyy", "fzz", "fx_exact", "fy_exact", "fz_exact", "fxx_exact", "fyy_exact", "fzz_exact"]
    data = [fx, fy, fz, fxx, fyy, fzz, fx_exact, fy_exact, fz_exact, fxx_exact, fyy_exact, fzz_exact]
#    data_t = [fx_t, fy_t, fz_t, fxx_t, fyy_t, fzz_t, fx_t_exact, fy_t_exact, fz_t_exact, fxx_t_exact, fyy_t_exact, fzz_t_exact]

    for t_i in range(L):

        t = t_i * dt

        # Only one f component
        f_t = f[:, t_i, 0:1]

        # Space derivatives
        fx_t = derivative(f_t, xyzt, create=True)[:, t_i, 0:1]
        fy_t = derivative(f_t, xyzt, create=True)[:, t_i, 1:2]
        fz_t = derivative(f_t, xyzt, create=True)[:, t_i, 2:3]
        fxx_t = derivative(fx_t, xyzt, create=False)[:, t_i, 0:1]
        fyy_t = derivative(fy_t, xyzt, create=False)[:, t_i, 1:2]
        fzz_t = derivative(fz_t, xyzt, create=False)[:, t_i, 2:3]

        fx_t = fx_t.detach().cpu().numpy().squeeze(-1)	# since the output is only f (not f=(f_1,f_2,...,f_n)), the format of f is [N, L, 1], and thus, eliminate the last dimension
        fy_t = fy_t.detach().cpu().numpy().squeeze(-1)
        fz_t = fz_t.detach().cpu().numpy().squeeze(-1)
        fxx_t = fxx_t.detach().cpu().numpy().squeeze(-1)
        fyy_t = fyy_t.detach().cpu().numpy().squeeze(-1)
        fzz_t = fzz_t.detach().cpu().numpy().squeeze(-1)

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

        data_t = [fx_t, fy_t, fz_t, fxx_t, fyy_t, fzz_t, fx_t_exact, fy_t_exact, fz_t_exact, fxx_t_exact, fyy_t_exact, fzz_t_exact]

        for j, k in zip(data, data_t):
            j.append(k)

    for i, j in zip(names, data):
        j = np.stack(j, axis=1)
        np.savetxt(f'GRADIENTS/{i}_ep{epoch}.csv', j, delimiter=',')


# INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER ############################

model = LSTMModel(input_dim=H_in, hidden_dim=7, layer_dim=2, output_dim=H_out)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# TRAIN THE LSTM MODEL #####################################################

num_epochs = 3
stepoch = 1
num_losses = int(num_epochs / stepoch)
loss_history = np.zeros((num_losses,), dtype=np.float64)
loss_i = 0

for epoch in range(num_epochs):

    h0, c0 = None, None

    model.train()		# Set the model in training mode
    optimizer.zero_grad()	# Clear old gradients (from previous epoch)

    # 3
#    print("xyzt_train", xyzt_train.shape, xyzt_train.device, xyzt_train.requires_grad, xyzt_train.is_leaf, xyzt_train.grad)
#    print("f_train_exact", f_train_exact.shape, f_train_exact.device, f_train_exact.requires_grad, f_train_exact.is_leaf, f_train_exact.grad)
#    xyzt_train_ = xyzt_train.detach().cpu().numpy()
#    f_train_exact_ = f_train_exact.squeeze(-1).detach().cpu().numpy()
#    np.savetxt(f'x_train_ep{epoch}.csv', xyzt_train_[:, :, 0], delimiter=',')
#    np.savetxt(f'y_train_ep{epoch}.csv', xyzt_train_[:, :, 1], delimiter=',')
#    np.savetxt(f'z_train_ep{epoch}.csv', xyzt_train_[:, :, 2], delimiter=',')
#    np.savetxt(f't_train_ep{epoch}.csv', xyzt_train_[:, :, 3], delimiter=',')
#    np.savetxt(f'f_train_exact_ep{epoch}.csv', f_train_exact_, delimiter=',')
#    if epoch == num_epochs-1: exit()
    #######

    # 10
    # this is needed for the gradients calculation (comment if not computing gradients)
    xyzt_train = xyzt_train.detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False): f_train_pred, h0, c0 = model(xyzt_train, epoch, h0, c0)
    #######

    # Pass variables to "forward()" function
    # Pass input "trainX", current model's "h0" and "c0"
    # Returns model predictions (f, "output"), and current model's "hn" and "cn" as the next model's (next epoch) "h0" and "c0"
    ##f_train_pred, h0, c0 = model(xyzt_train, epoch, h0, c0)

    # 8
    # compute gradients (not needed)
#    print("f_train_pred", f_train_pred.shape, f_train_pred.device, f_train_pred.requires_grad, f_train_pred.is_leaf, f_train_pred.grad)
#    print("XYZT_TRAIN", xyzt_train.shape)
    if epoch == num_epochs - 1: compute_gradients(f_train_pred, xyzt_train, epoch, L, dt)
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

# 11
# this is needed for the gradients calculation (comment if not computing gradients)
xyzt_test = xyzt_test.detach().requires_grad_(True)
with torch.backends.cudnn.flags(enabled=False): f_test_pred, _, _ = model(xyzt_test, epoch, h0, c0)	# cuDNN is not compatible with double backward pass (2nd derivatives) in LSTMs
#######

##with torch.no_grad(): f_test_pred, _, _ = model(xyzt_test, epoch, h0, c0)	# Predict y_pred in the test set. No need for gradients.

# 9
# compute gradients (not needed)
print("f_test_pred", f_test_pred.shape, f_test_pred.device, f_test_pred.requires_grad, f_test_pred.is_leaf, f_test_pred.grad)
print("XYZT_TEST", xyzt_test.shape)
compute_gradients(f_test_pred, xyzt_test, epoch, L, dt)
#######

f_test_pred = f_test_pred.squeeze(-1).detach().cpu().numpy()
f_test_exact = f_test_exact.squeeze(-1).detach().cpu().numpy()
#np.savetxt('f_test_pred.csv', f_test_pred, delimiter=',')
#np.savetxt('f_test_exact.csv', f_test_exact, delimiter=',')

l2_error = np.linalg.norm(f_test_exact - f_test_pred) / np.linalg.norm(f_test_exact)
print("L2 error:", l2_error)


