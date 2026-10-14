import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


# Defining GPU device as "device"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# Precision
precision = torch.float64

np.random.seed(0)
torch.manual_seed(0)

N_interior = 200		# number of (x,y,z,t) training examples (interior)
N_face = 50	# number of (x,y,z,t) training examples per boundary plane/face
N = N_interior + 6 * N_face	# total number of training examples
H_in = 4		# number of input features
H_out = 4		# number of output features
L = 101			# sequence length (number of time steps)
n_test = 16		# number of test points per space coordinate
N_test_cub = n_test**3	# number of (x,y,z,t) test examples
#pi = torch.pi
#cos = torch.cos
x_min = torch.tensor([-1, -1, -1], device=device, dtype=precision)
x_max = torch.tensor([ 1, 1, 1], device=device, dtype=precision)
t_min, t_max = 0, 1
dt = (t_max-t_min)/(L-1)
a, d, Re = 1, 1, 1
loss_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device=device, dtype=precision)
n_bc = 3      # number of BC loss components
n_ic = 3      # number of IC loss components
n_res = 4       # number of residual loss components
n_loss = n_bc + n_ic + n_res		# total number of loss components

# BELTRAMI FLOW ANALYTICAL SOLUTIONS #######################################

# u_func, v_func, w_func, p_func: analytical solutions of the Beltrami equations
# for any time instant

def u_func(xyzt):
    return (
        -a
        * (
            torch.exp(a * xyzt[:, :, 0:1]) * torch.sin(a * xyzt[:, :, 1:2] + d * xyzt[:, :, 2:3])
            + torch.exp(a * xyzt[:, :, 2:3]) * torch.cos(a * xyzt[:, :, 0:1] + d * xyzt[:, :, 1:2])
        )
        * torch.exp(-(d ** 2) * xyzt[:, :, 3:4])
    )


def v_func(xyzt):
    return (
        -a
        * (
            torch.exp(a * xyzt[:, :, 1:2]) * torch.sin(a * xyzt[:, :, 2:3] + d * xyzt[:, :, 0:1])
            + torch.exp(a * xyzt[:, :, 0:1]) * torch.cos(a * xyzt[:, :, 1:2] + d * xyzt[:, :, 2:3])
        )
        * torch.exp(-(d ** 2) * xyzt[:, :, 3:4])
    )


def w_func(xyzt):
    return (
        -a
        * (
            torch.exp(a * xyzt[:, :, 2:3]) * torch.sin(a * xyzt[:, :, 0:1] + d * xyzt[:, :, 1:2])
            + torch.exp(a * xyzt[:, :, 1:2]) * torch.cos(a * xyzt[:, :, 2:3] + d * xyzt[:, :, 0:1])
        )
        * torch.exp(-(d ** 2) * xyzt[:, :, 3:4])
    )

def p_func(xyzt):
    return (
        -0.5
        * a ** 2
        * (
            torch.exp(2 * a * xyzt[:, :, 0:1])
            + torch.exp(2 * a * xyzt[:, :, 1:2])
            + torch.exp(2 * a * xyzt[:, :, 2:3])
            + 2
            * torch.sin(a * xyzt[:, :, 0:1] + d * xyzt[:, :, 1:2])
            * torch.cos(a * xyzt[:, :, 2:3] + d * xyzt[:, :, 0:1])
            * torch.exp(a * (xyzt[:, :, 1:2] + xyzt[:, :, 2:3]))
            + 2
            * torch.sin(a * xyzt[:, :, 1:2] + d * xyzt[:, :, 2:3])
            * torch.cos(a * xyzt[:, :, 0:1] + d * xyzt[:, :, 1:2])
            * torch.exp(a * (xyzt[:, :, 2:3] + xyzt[:, :, 0:1]))
            + 2
            * torch.sin(a * xyzt[:, :, 2:3] + d * xyzt[:, :, 0:1])
            * torch.cos(a * xyzt[:, :, 1:2] + d * xyzt[:, :, 2:3])
            * torch.exp(a * (xyzt[:, :, 0:1] + xyzt[:, :, 1:2]))
        )
        * torch.exp(-2 * d ** 2 * xyzt[:, :, 3:4])
    )


# PREPARE DATA #############################################################

# Training data, inside (random x,y,z; sequential t)
xyz_train = x_min + (x_max - x_min) * torch.rand(N_interior, H_in-1, device=device, dtype=precision)	# random (x,y,z) coordinates in [x_min, x_max] for the training set
xyz_train_rep = xyz_train.unsqueeze(1).repeat(1, L, 1)					# (N_interior,H_in-1) -> (N_interior,1,H_in-1) -> (N_interior,L,H_in-1)
t_train = torch.linspace(t_min, t_max, L, dtype=precision, device=device)			# t-coordinate [0,1] in steps of 0.01
t_train_rep = t_train.unsqueeze(0).unsqueeze(-1).repeat(N_interior, 1, 1)			# (L,) -> (1,L) -> (1,L,1) -> (N_interior,L,1)
xyzt_train_interior = torch.cat((xyz_train_rep, t_train_rep), dim=2)				# torch.cat() joins a sequence of tensors along an *existing* dimension (torch.stack() would be along a *new* dimension)
#uvwp_train_exact_interior = torch.cat((u_func(xyzt_train_interior), v_func(xyzt_train_interior), w_func(xyzt_train_interior), p_func(xyzt_train_interior)), dim=2)

# Training data, boundaries (random x,y,z; sequential t)
xz = x_min[[0, 2]] + (x_max[[0, 2]] - x_min[[0, 2]]) * torch.rand(N_face, 2, device=device, dtype=precision)	# (N_face, H_in-2); H_in-2 = xz
yz = x_min[[1, 2]] + (x_max[[1, 2]] - x_min[[1, 2]]) * torch.rand(N_face, 2, device=device, dtype=precision)
xy = x_min[[0, 1]] + (x_max[[0, 1]] - x_min[[0, 1]]) * torch.rand(N_face, 2, device=device, dtype=precision)
xz_top = torch.stack((xz[:, 0], x_max[1].expand(N_face), xz[:, 1]), dim=1)	# Add the fixed coordinate: (N_face, H_in-1)
xz_bottom = torch.stack((xz[:, 0], x_min[1].expand(N_face), xz[:, 1]), dim=1)
yz_left = torch.stack((x_min[0].expand(N_face), yz[:, 0], yz[:, 1]), dim=1)
yz_right = torch.stack((x_max[0].expand(N_face), yz[:, 0], yz[:, 1]), dim=1)
xy_front = torch.stack((xy[:, 0], xy[:, 1], x_max[2].expand(N_face)), dim=1)
xy_back = torch.stack((xy[:, 0], xy[:, 1], x_min[2].expand(N_face)), dim=1)
xyz_boundaries = torch.cat((xz_top, xz_bottom, yz_left, yz_right, xy_front, xy_back), dim=0)	# Concatenate the faces: (6*N_face, H_in-1)
xyz_boundaries_rep = xyz_boundaries.unsqueeze(1).repeat(1, L, 1)	# Repeat the space coordinates throughout the time steps: (6*N_face, L, H_in-1)
t_rep = t_train.unsqueeze(0).unsqueeze(-1).repeat(6 * N_face, 1, 1)	# Repeat the time coordinate vector '6*N_face' times and add a 3rd dimension: (6*N_face, L, 1)
xyzt_boundaries = torch.cat((xyz_boundaries_rep, t_rep), dim=2)		# Concatenate the space coordinates with the time coordinate: (6*N_face, L, H_in)

# Complete training data (inside + boundaries)
xyzt_train = torch.cat((xyzt_train_interior, xyzt_boundaries), dim=0)		# Concatenate the interior with the boundary training points to form the full training set: (N, L, H_in)
uvwp_train_exact = torch.cat((u_func(xyzt_train), v_func(xyzt_train), w_func(xyzt_train), p_func(xyzt_train)), dim=2)	# Exact (u,v,w,p) for the training points: (N, L, H_out)

##
#print("xyz_boundaries", xyz_boundaries.shape)
#np.savetxt('xyz_boundaries.csv', xyz_boundaries.detach().cpu().numpy(), delimiter=',')
#print("xyz_boundaries_rep", xyz_boundaries_rep.shape)
#for i in range(L):
#    np.savetxt(f'xyz_boundaries_rep_{i}.csv', xyz_boundaries_rep[:,i,:].detach().cpu().numpy(), delimiter=',')
#print("t_rep", t_rep.shape)
#print("xyzt_boundaries", xyzt_boundaries.shape)
#np.savetxt('t_rep.csv', t_rep.squeeze(-1).detach().cpu().numpy(), delimiter=',')
#print("xyzt_boundaries", xyzt_boundaries.shape, xyzt_boundaries.device)
#for i in range(L):
#    np.savetxt(f'xyzt_boundaries_{i}.csv', xyzt_boundaries[:,i,:].detach().cpu().numpy(), delimiter=',')
#print("xyzt_train", xyzt_train.shape, xyzt_train.device)
#print("uvwp_train_exact", uvwp_train_exact.shape, uvwp_train_exact.device)
#for i in range(L):
#    np.savetxt(f'xyzt_train_{i}.csv', xyzt_train[:,i,:].detach().cpu().numpy(), delimiter=',')
#    np.savetxt(f'xyzt_train_int_{i}.csv', xyzt_train[:N_interior,i,:].detach().cpu().numpy(), delimiter=',')
#    np.savetxt(f'uvwp_train_exact_interior_{i}.csv', uvwp_train_exact_interior[:,i,:].detach().cpu().numpy(), delimiter=',')
#    np.savetxt(f'uvwp_train_exact_{i}.csv', uvwp_train_exact[:N_interior,i,:].detach().cpu().numpy(), delimiter=',')
#print("uvwp_train_exact", uvwp_train_exact.shape, uvwp_train_exact.device)
#exit()
##

#print("p_func", p_func(xyzt_train).shape)
#np.savetxt('p_func.csv', p_func(xyzt_train).squeeze(-1).detach().cpu().numpy(), delimiter=',')
#print("uvwp_train_exact", uvwp_train_exact.shape)
#res = p_func(xyzt_train) * 2
#for i in range(N):
#    np.savetxt(f'xyzt_train_{i}.csv', xyzt_train[i:i+1, :, :].squeeze(0).detach().cpu().numpy(), delimiter=',')
#np.savetxt('p_func.csv', p_func(xyzt_train).squeeze(-1).detach().cpu().numpy(), delimiter=',')
#np.savetxt('res.csv', res.squeeze(-1).detach().cpu().numpy(), delimiter=',')
#print("xyzt_train", xyzt_train.shape)
#exit()
#x, y, z, t = xyzt_train[:,:,0], xyzt_train[:,:,1], xyzt_train[:,:,2], xyzt_train[:,:,3]	# Each coordinate corresponds to a (i,j) plan
#f_train_exact = (x**3 + 4 * t * x * y**2 + cos(2 * z * t)).unsqueeze(-1)			# output training set: (N,L) -> (N,L,1
#np.savetxt("xyz_train.csv", xyz_train.detach().cpu().numpy(), delimiter=',')

# Test data (structured grid x,y,z; sequential t)
x, y, z = torch.meshgrid(
          torch.linspace(x_min[0], x_max[0], n_test, dtype=precision, device=device),
          torch.linspace(x_min[1], x_max[1], n_test, dtype=precision, device=device),
          torch.linspace(x_min[2], x_max[2], n_test, dtype=precision, device=device),
          indexing='ij')								# Create a N_test_cub mesh, with uniformly spaced mesh points
xyz_test = torch.vstack((torch.ravel(x), torch.ravel(y), torch.ravel(z))).T		# Matrix (N_test_cub,3): each line is the (x,y,z) coordinates of a mesh point
xyz_test_rep = xyz_test.unsqueeze(1).repeat(1, L, 1)                  			# (N_test_cub,H_in-1) -> (N_test_cub,1,H_in-1) -> (N_test_cub,L,H_in-1)
t_test = torch.linspace(t_min, t_max, L, dtype=precision, device=device)                        # t-coordinate [0,1] in steps of 0.01
t_test_rep = t_test.unsqueeze(0).unsqueeze(-1).repeat(N_test_cub, 1, 1)                 # (L,) -> (1,L) -> (1,L,1) -> (N_test_cub,L,1)
xyzt_test = torch.cat((xyz_test_rep, t_test_rep), dim=2)
uvwp_test_exact = torch.cat((u_func(xyzt_test), v_func(xyzt_test), w_func(xyzt_test), p_func(xyzt_test)), dim=2)

#print("uvwp_test_exact", uvwp_test_exact.shape)
#res = p_func(xyzt_test) * 2
#test_set_ = []
#for i in range(N_test_cub):
#    test_set_.append(xyzt_test[i:i+1, :, :].squeeze(0))
#test_set = torch.cat(test_set_, dim=0)
#print("xyzt_test", xyzt_test.shape)
#np.savetxt('xyzt_test.csv', test_set.squeeze(0).detach().cpu().numpy(), delimiter=',')
#np.savetxt('p_func.csv', p_func(xyzt_test).squeeze(-1).detach().cpu().numpy(), delimiter=',')
#np.savetxt('res.csv', res.squeeze(-1).detach().cpu().numpy(), delimiter=',')
#exit()
#x, y, z, t = xyzt_test[:,:,0], xyzt_test[:,:,1], xyzt_test[:,:,2], xyzt_test[:,:,3] 	# Each coordinate corresponds to a (i,j) plan
#f_test_exact = (x**3 + 4 * t * x * y**2 + cos(2 * z * t)).unsqueeze(-1)                	# output test set: (N_test_cub,L) -> (N_test_cub,L,1)
#np.savetxt("xyz_test.csv", xyz_test.detach().cpu().numpy(), delimiter=',')


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
    def forward(self, xyzt, epoch, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, xyzt.size(0), self.hidden_dim, dtype=precision, device=device)	# initial hidden state
            c0 = torch.zeros(self.layer_dim, xyzt.size(0), self.hidden_dim, dtype=precision, device=device)	# initial cell state

        out, (hn, cn) = self.lstm(xyzt, (h0, c0))	# out = hidden-state output at every sequence element
        out = self.fc(out)	# call to prediction: for all time steps, y=W*h+b, where W and b are the parameters of the linear layer

        return out, hn, cn


# COMPUTE DERIVATIVES ######################################################

# not needed for this model; this is just to see if the derivatives of f are well calculated

def compute_gradients(uvwp, xyzt, L, dt, epoch, model):

    # Derivatives through finite differences
    def finite_differences(uvwp, xyzt, L, dt, epoch, model):

        epsilon = 1e-4              # small number (but not too small)

        fd_deriv_i = []
        fd_xyzt = xyzt[:N_interior, :, :]
        x, y, z, t = fd_xyzt[:,:,0:1], fd_xyzt[:,:,1:2], fd_xyzt[:,:,2:3], fd_xyzt[:,:,3:4]

        #with torch.no_grad(): fd_uvwp, _, _ = model(fd_xyzt, epoch, None, None)
        fd_uvwp, _, _ = model(fd_xyzt, epoch, None, None)

        for i in range(3):  # 3 = (x,y,z)

            fd_xyzt_plus = fd_xyzt.clone()
            fd_xyzt_minus = fd_xyzt.clone()

            fd_xyzt_plus[:, :, i] += epsilon        # x = fd_xyzt[:, :, 0]; y = fd_xyzt[:, :, 1]; z = fd_xyzt[:, :, 2];
            fd_xyzt_minus[:, :, i] -= epsilon

            xp, yp, zp, tp = fd_xyzt_plus[:,:,0:1], fd_xyzt_plus[:,:,1:2], fd_xyzt_plus[:,:,2:3], fd_xyzt_plus[:,:,3:4]
            xm, ym, zm, tm = fd_xyzt_minus[:,:,0:1], fd_xyzt_minus[:,:,1:2], fd_xyzt_minus[:,:,2:3], fd_xyzt_minus[:,:,3:4]

            #with torch.no_grad():
            #    fd_uvwp_plus, _, _ = model(fd_xyzt_plus, epoch, None, None)
            #    fd_uvwp_minus, _, _ = model(fd_xyzt_minus, epoch, None, None)
            fd_uvwp_plus, _, _ = model(fd_xyzt_plus, epoch, None, None)
            fd_uvwp_minus, _, _ = model(fd_xyzt_minus, epoch, None, None)

            # 1st and 2nd derivatives: central diferences
            for j in range(4):              # 4 = (u,v,w,p)
                j_i_fd = (fd_uvwp_plus[:, :, j:j+1] - fd_uvwp_minus[:, :, j:j+1]) / (2 * epsilon)
                if j <= 2:
                    j_ii_fd = (fd_uvwp_plus[:, :, j:j+1] - 2 * fd_uvwp[:, :, j:j+1] + fd_uvwp_minus[:, :, j:j+1]) / (epsilon**2)
                    j_i_12fd = torch.cat((j_i_fd, j_ii_fd), dim=2)
                else:
                    j_i_12fd = j_i_fd
                fd_deriv_i.append(j_i_12fd)

            fd_deriv = torch.cat(fd_deriv_i, dim=2)

        return fd_deriv

    def autograd(uvwp, xyzt, L, dt, epoch, model):

        def derivative(aa, bb, create=True):
            return torch.autograd.grad(aa, bb, grad_outputs=torch.ones_like(aa), create_graph=create)[0]

        grad_uvwp = []
        grad_deriv = []

        # create=True even for the highest order derivatives for (u,v,w,p) since dL/dθ is also composed of the derivatives of them (chain rule)

        for t_n1 in range(L):

            t = t_n1 * dt

            # (u,v,w,p) components at current time step
            u_n1 = uvwp[:N_interior, t_n1, 0:1]
            v_n1 = uvwp[:N_interior, t_n1, 1:2]
            w_n1 = uvwp[:N_interior, t_n1, 2:3]
            p_n1 = uvwp[:N_interior, t_n1, 3:4]

            # Space derivatives - u
            grad_u = derivative(u_n1, xyzt, create=True)
            u_x_all = grad_u[:N_interior, :t_n1+1, 0:1]
            u_y_all = grad_u[:N_interior, :t_n1+1, 1:2]
            u_z_all = grad_u[:N_interior, :t_n1+1, 2:3]
            u_xx_all = derivative(u_x_all, xyzt, create=True)[:N_interior, :t_n1+1, 0:1]
            u_yy_all = derivative(u_y_all, xyzt, create=True)[:N_interior, :t_n1+1, 1:2]
            u_zz_all = derivative(u_z_all, xyzt, create=True)[:N_interior, :t_n1+1, 2:3]
            u_x_n1 = torch.sum(u_x_all, 1)
            u_y_n1 = torch.sum(u_y_all, 1)
            u_z_n1 = torch.sum(u_z_all, 1)
            u_xx_n1 = torch.sum(u_xx_all, 1)
            u_yy_n1 = torch.sum(u_yy_all, 1)
            u_zz_n1 = torch.sum(u_zz_all, 1)

            # Space derivatives - v
            grad_v = derivative(v_n1, xyzt, create=True)
            v_x_all = grad_v[:N_interior, :t_n1+1, 0:1]
            v_y_all = grad_v[:N_interior, :t_n1+1, 1:2]
            v_z_all = grad_v[:N_interior, :t_n1+1, 2:3]
            v_xx_all = derivative(v_x_all, xyzt, create=True)[:N_interior, :t_n1+1, 0:1]
            v_yy_all = derivative(v_y_all, xyzt, create=True)[:N_interior, :t_n1+1, 1:2]
            v_zz_all = derivative(v_z_all, xyzt, create=True)[:N_interior, :t_n1+1, 2:3]
            v_x_n1 = torch.sum(v_x_all, 1)
            v_y_n1 = torch.sum(v_y_all, 1)
            v_z_n1 = torch.sum(v_z_all, 1)
            v_xx_n1 = torch.sum(v_xx_all, 1)
            v_yy_n1 = torch.sum(v_yy_all, 1)
            v_zz_n1 = torch.sum(v_zz_all, 1)

            # Space derivatives - w
            grad_w = derivative(w_n1, xyzt, create=True)
            w_x_all = grad_w[:N_interior, :t_n1+1, 0:1]
            w_y_all = grad_w[:N_interior, :t_n1+1, 1:2]
            w_z_all = grad_w[:N_interior, :t_n1+1, 2:3]
            w_xx_all = derivative(w_x_all, xyzt, create=True)[:N_interior, :t_n1+1, 0:1]
            w_yy_all = derivative(w_y_all, xyzt, create=True)[:N_interior, :t_n1+1, 1:2]
            w_zz_all = derivative(w_z_all, xyzt, create=True)[:N_interior, :t_n1+1, 2:3]
            w_x_n1 = torch.sum(w_x_all, 1)
            w_y_n1 = torch.sum(w_y_all, 1)
            w_z_n1 = torch.sum(w_z_all, 1)
            w_xx_n1 = torch.sum(w_xx_all, 1)
            w_yy_n1 = torch.sum(w_yy_all, 1)
            w_zz_n1 = torch.sum(w_zz_all, 1)

            # Space derivatives - p
            grad_p = derivative(p_n1, xyzt, create=True)
            p_x_all = grad_p[:N_interior, :t_n1+1, 0:1]
            p_y_all = grad_p[:N_interior, :t_n1+1, 1:2]
            p_z_all = grad_p[:N_interior, :t_n1+1, 2:3]
            p_x_n1 = torch.sum(p_x_all, 1)
            p_y_n1 = torch.sum(p_y_all, 1)
            p_z_n1 = torch.sum(p_z_all, 1)

            grad_deriv_x_i = torch.stack((u_x_n1, u_xx_n1, v_x_n1, v_xx_n1, w_x_n1, w_xx_n1, p_x_n1), dim=2)
            grad_deriv_y_i = torch.stack((u_y_n1, u_yy_n1, v_y_n1, v_yy_n1, w_y_n1, w_yy_n1, p_y_n1), dim=2)
            grad_deriv_z_i = torch.stack((u_z_n1, u_zz_n1, v_z_n1, v_zz_n1, w_z_n1, w_zz_n1, p_z_n1), dim=2)
            grad_deriv_i = torch.cat((grad_deriv_x_i, grad_deriv_y_i, grad_deriv_z_i), dim=2)
            grad_deriv.append(grad_deriv_i)

        grad_deriv = torch.cat(grad_deriv, dim=1)   # (N_interior, L, total nr of derivatives)

        return grad_deriv

###########

#    torch.cuda.synchronize()
#    start1 = time.perf_counter()

    fd_deriv = finite_differences(uvwp, xyzt, L, dt, epoch, model)

#    torch.cuda.synchronize()
#    end1 = time.perf_counter()
#    print(f"Elapsed time: {end1 - start1:.6f} s")

#    grad_deriv = autograd(uvwp, xyzt, L, dt, epoch, model)
###############*****************

    deriv = fd_deriv
#    deriv = grad_deriv
#    print("deriv", deriv.requires_grad)

    # List to accumulate the result for each time step
    uvwp_grad_mid_all = []

    for t_n1 in range(L):

        # (u,v,w,p) components at current time step
        u_n1 = uvwp[:N_interior, t_n1, 0:1]
        v_n1 = uvwp[:N_interior, t_n1, 1:2]
        w_n1 = uvwp[:N_interior, t_n1, 2:3]
        p_n1 = uvwp[:N_interior, t_n1, 3:4]

        if t_n1 >= 1:

            # Midpoint velocities and pressure
            u_mid = 0.5 * (u_n1 + u_n)
            v_mid = 0.5 * (v_n1 + v_n)
            w_mid = 0.5 * (w_n1 + w_n)
            p_mid = 0.5 * (p_n1 + p_n)

            # Time derivatives at midpoint
            u_t_mid = (u_n1 - u_n) / dt
            v_t_mid = (v_n1 - v_n) / dt
            w_t_mid = (w_n1 - w_n) / dt

            # Midpoint derivatives - u
            u_x_mid = 0.5 * (deriv[:,t_n1,0:1] + deriv[:,t_n1-1,0:1])
            u_y_mid = 0.5 * (deriv[:,t_n1,7:8] + deriv[:,t_n1-1,7:8])
            u_z_mid = 0.5 * (deriv[:,t_n1,14:15] + deriv[:,t_n1-1,14:15])
            u_xx_mid = 0.5 * (deriv[:,t_n1,1:2] + deriv[:,t_n1-1,1:2])
            u_yy_mid = 0.5 * (deriv[:,t_n1,8:9] + deriv[:,t_n1-1,8:9])
            u_zz_mid = 0.5 * (deriv[:,t_n1,15:16] + deriv[:,t_n1-1,15:16])

            # Midpoint derivatives - v
            v_x_mid = 0.5 * (deriv[:,t_n1,2:3] + deriv[:,t_n1-1,2:3])
            v_y_mid = 0.5 * (deriv[:,t_n1,9:10] + deriv[:,t_n1-1,9:10])
            v_z_mid = 0.5 * (deriv[:,t_n1,16:17] + deriv[:,t_n1-1,16:17])
            v_xx_mid = 0.5 * (deriv[:,t_n1,3:4] + deriv[:,t_n1-1,3:4])
            v_yy_mid = 0.5 * (deriv[:,t_n1,10:11] + deriv[:,t_n1-1,10:11])
            v_zz_mid = 0.5 * (deriv[:,t_n1,17:18] + deriv[:,t_n1-1,17:18])

            # Midpoint derivatives - w
            w_x_mid = 0.5 * (deriv[:,t_n1,4:5] + deriv[:,t_n1-1,4:5])
            w_y_mid = 0.5 * (deriv[:,t_n1,11:12] + deriv[:,t_n1-1,11:12])
            w_z_mid = 0.5 * (deriv[:,t_n1,18:19] + deriv[:,t_n1-1,18:19])
            w_xx_mid = 0.5 * (deriv[:,t_n1,5:6] + deriv[:,t_n1-1,5:6])
            w_yy_mid = 0.5 * (deriv[:,t_n1,12:13] + deriv[:,t_n1-1,12:13])
            w_zz_mid = 0.5 * (deriv[:,t_n1,19:20] + deriv[:,t_n1-1,19:20])

            # Midpoint derivatives - p
            p_x_mid = 0.5 * (deriv[:,t_n1,6:7] + deriv[:,t_n1-1,6:7])
            p_y_mid = 0.5 * (deriv[:,t_n1,13:14] + deriv[:,t_n1-1,13:14])
            p_z_mid = 0.5 * (deriv[:,t_n1,20:21] + deriv[:,t_n1-1,20:21])

            uvwp_mid = torch.stack((u_mid, v_mid, w_mid, p_mid), dim=2)
            grad_u_mid = torch.stack((u_t_mid, u_x_mid, u_y_mid, u_z_mid, u_xx_mid, u_yy_mid, u_zz_mid), dim=2)
            grad_v_mid = torch.stack((v_t_mid, v_x_mid, v_y_mid, v_z_mid, v_xx_mid, v_yy_mid, v_zz_mid), dim=2)
            grad_w_mid = torch.stack((w_t_mid, w_x_mid, w_y_mid, w_z_mid, w_xx_mid, w_yy_mid, w_zz_mid), dim=2)
            grad_p_mid = torch.stack((p_x_mid, p_y_mid, p_z_mid), dim=2)
            uvwp_grad_mid = torch.cat((uvwp_mid, grad_u_mid, grad_v_mid, grad_w_mid, grad_p_mid), dim=2)
            uvwp_grad_mid_all.append(uvwp_grad_mid)

        u_n, v_n, w_n, p_n = u_n1, v_n1, w_n1, p_n1
#################**************

    uvwp_grad_mid_all = torch.cat(uvwp_grad_mid_all, dim=1)

    def compare_derivatives(grad_deriv, fd_deriv):
        grad_deriv = grad_deriv.detach().cpu().numpy()          # derivatives from Autograd
        fd_deriv = fd_deriv.cpu().numpy()                       # derivatives from Finite Differences

        l2_error_1 = np.linalg.norm(fd_deriv - grad_deriv) / np.linalg.norm(fd_deriv)
        print(f"\nL2 error between Autograd and Finite Differences (epoch {epoch}):", l2_error_1, "\n")

#    compare_derivatives(grad_deriv, fd_deriv)

    return uvwp_grad_mid_all


# NAVIER-STOKES EQUATIONS ##################################################

#print("uvwp_train_pred", uvwp_train_exact[:N_interior, 1:, :].shape)
#exit()

# HERE :: mom_x, mom_y, mom_z, cont = pde(uvwp_train_pred_res, xyz_train_res)

def pde(u, xyzt, Re, epoch, model):

    grads = compute_gradients(u, xyzt, L, dt, epoch, model)

    F_mid_mx = (
        (grads[:,:,0:1] * grads[:,:,5:6] + grads[:,:,1:2] * grads[:,:,6:7] + grads[:,:,2:3] * grads[:,:,7:8])
        + grads[:,:,25:26]
        - 1 / Re * (grads[:,:,8:9] + grads[:,:,9:10] + grads[:,:,10:11])
    )

    F_mid_my = (
        (grads[:,:,0:1] * grads[:,:,12:13] + grads[:,:,1:2] * grads[:,:,13:14] + grads[:,:,2:3] * grads[:,:,14:15])
        + grads[:,:,26:27]
        - 1 / Re * (grads[:,:,15:16] + grads[:,:,16:17] + grads[:,:,17:18])
    )

    F_mid_mz = (
        (grads[:,:,0:1] * grads[:,:,19:20] + grads[:,:,1:2] * grads[:,:,20:21] + grads[:,:,2:3] * grads[:,:,21:22])
        + grads[:,:,27:28]
        - 1 / Re * (grads[:,:,22:23] + grads[:,:,23:24] + grads[:,:,24:25])
    )

    momentum_x = grads[:,:,4:5] + F_mid_mx
    momentum_y = grads[:,:,11:12] + F_mid_my
    momentum_z = grads[:,:,18:19] + F_mid_mz
    continuity = grads[:,:,5:6] + grads[:,:,13:14] + grads[:,:,21:22]

    return momentum_x, momentum_y, momentum_z, continuity


# INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER ############################

model = LSTMModel(input_dim=H_in, hidden_dim=100, layer_dim=20, output_dim=H_out)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# TRAIN THE LSTM MODEL #####################################################

num_epochs = 1000
stepoch = 1
num_losses = int(num_epochs / stepoch)
loss_history = np.zeros((num_losses, len(loss_weights)), dtype=np.float64)
loss_i = 0

print("\nEPOCH      TRAINING LOSSES\n")

def bc_ic_loss(n_bc, n_ic, criterion, uvwp_train_pred, uvwp_train_exact):

    loss_bc = []
    loss_ic = []

    # BC losses
    for i in range(n_bc):
        uvwp_train_pred_bc = uvwp_train_pred[N_interior:, :, i]
        uvwp_train_exact_bc = uvwp_train_exact[N_interior:, :, i]
        loss_bc_i = criterion(uvwp_train_pred_bc, uvwp_train_exact_bc)
        loss_bc.append(loss_bc_i)
    loss_bc = torch.stack(loss_bc)

    # IC losses
    for i in range(n_ic):
        uvwp_train_pred_ic = uvwp_train_pred[:N_interior, 0, i]
        uvwp_train_exact_ic = uvwp_train_exact[:N_interior, 0, i]
        loss_ic_i = criterion(uvwp_train_pred_ic, uvwp_train_exact_ic)
        loss_ic.append(loss_ic_i)
    loss_ic = torch.stack(loss_ic)

    # Concatenate BC and IC loss vectors
    loss_bcic = torch.cat((loss_bc, loss_ic))	# unweighted BC and IC losses

    return loss_bcic

def res_loss(Re, criterion, model, loss_weights, uvwp_train_pred, xyzt_train, epoch):

    momentum_x, momentum_y, momentum_z, continuity = pde(uvwp_train_pred, xyzt_train, Re, epoch, model)	   # compute residuals

    loss_momentum_x = criterion(momentum_x, torch.zeros_like(momentum_x))
    loss_momentum_y = criterion(momentum_y, torch.zeros_like(momentum_y))
    loss_momentum_z = criterion(momentum_z, torch.zeros_like(momentum_z))
    loss_continuity = criterion(continuity, torch.zeros_like(continuity))

    loss_residuals = torch.stack((loss_momentum_x, loss_momentum_y, loss_momentum_z, loss_continuity), dim=0)

    return loss_residuals


for epoch in range(num_epochs):

    # Wall-clock time counter: start
    torch.cuda.synchronize()
    start = time.perf_counter()

    h0, c0 = None, None

    model.train()		# Set the model in training mode
    optimizer.zero_grad()	# Clear old gradients (from previous epoch)

    # 10
    # this is needed for the gradients calculation (comment if not computing gradients with AUTOGRAD)
    # COMMENT IF **NOT** COMPUTING GRADIENTS WITH AUTOGRAD
#    xyzt_train = xyzt_train.detach().requires_grad_(True)
#    with torch.backends.cudnn.flags(enabled=False): uvwp_train_pred, h0, c0 = model(xyzt_train, epoch, h0, c0)
    #######

    # Pass variables to "forward()" function
    # Pass input "trainX", current model's "h0" and "c0"
    # Returns model predictions (f, "output"), and current model's "hn" and "cn" as the next model's (next epoch) "h0" and "c0"
    # COMMENT IF COMPUTING GRADIENTS WITH AUTOGRAD
    uvwp_train_pred, h0, c0 = model(xyzt_train, epoch, h0, c0)

    loss_bcic = bc_ic_loss(n_bc, n_ic, criterion, uvwp_train_pred, uvwp_train_exact)
    loss_residuals = res_loss(Re, criterion, model, loss_weights, uvwp_train_pred, xyzt_train, epoch)
    unweighted_losses = torch.cat((loss_residuals, loss_bcic))
    losses = unweighted_losses * loss_weights
    total_loss = torch.sum(losses)

    # 8
    # compute gradients (not needed)
#    print("uvwp_train_pred", uvwp_train_pred.shape, uvwp_train_pred.device, uvwp_train_pred.requires_grad, uvwp_train_pred.is_leaf, uvwp_train_pred.grad)
#    print("XYZT_TRAIN", xyzt_train.shape)
#    if epoch == num_epochs: gradients = compute_gradients(uvwp_train_pred, xyzt_train, epoch, L, dt)
    #######

    # Backpropagation and parameter update
    total_loss.backward()			# Gradient of the loss w.r.t. trainable parameters (dL/dθ)
    optimizer.step()			# Updates the parameters: θ(i) = θ(i-1) - η * dL/dθ

    # Wall-clock time counter: end
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} s")

    unweighted_losses_print = unweighted_losses.detach().cpu().numpy()

    # Print epoch and corresponding loss value
    if (epoch + 1) % stepoch == 0:
        epoch_label = f"{epoch+1}/{num_epochs}"
        epoch_width = len(f"{num_epochs}/{num_epochs}")

        losses_str = np.array2string(unweighted_losses_print, formatter={"float_kind": lambda x: f"{x:.2e}"})

        print(f"{epoch_label:<{epoch_width}}   {losses_str}")

        loss_history[loss_i, :] = unweighted_losses_print
        loss_i += 1

np.savetxt('loss_history.csv', loss_history, delimiter=',')

print("\nEND OF TRAINING\n")
#exit()

# EVALUATE PREDICTIONS #####################################################

h0, c0 = None, None

epoch = 'TEST'

model.eval()    # Set the model in evaluation mode

#
#xyzt_test = xyzt_train
#uvwp_test_exact = uvwp_train_exact

# 11
# this is needed for the gradients calculation (comment if not computing gradients)
xyzt_test = xyzt_test.detach().requires_grad_(True)
with torch.backends.cudnn.flags(enabled=False): uvwp_test_pred, _, _ = model(xyzt_test, epoch, h0, c0)	# cuDNN is not compatible with double backward pass (2nd derivatives) in LSTMs
#######

#with torch.no_grad(): uvwp_test_pred, _, _ = model(xyzt_test, epoch, h0, c0)	# Predict y_pred in the test set. No need for gradients.

# 9
# compute gradients (not needed)
#print("uvwp_test_pred", uvwp_test_pred.shape, uvwp_test_pred.device, uvwp_test_pred.requires_grad, uvwp_test_pred.is_leaf, uvwp_test_pred.grad)
#print("XYZT_TEST", xyzt_test.shape)
#gradients = compute_gradients(uvwp_test_pred, xyzt_test, epoch, L, dt)
#######

print("uvwp_test_pred", uvwp_test_pred.shape)

u_test_pred = uvwp_test_pred[:, :, 0].detach().cpu().numpy()
v_test_pred = uvwp_test_pred[:, :, 1].detach().cpu().numpy()
w_test_pred = uvwp_test_pred[:, :, 2].detach().cpu().numpy()
p_test_pred = uvwp_test_pred[:, :, 3].detach().cpu().numpy()

u_test_exact = uvwp_test_exact[:, :, 0].detach().cpu().numpy()
v_test_exact = uvwp_test_exact[:, :, 1].detach().cpu().numpy()
w_test_exact = uvwp_test_exact[:, :, 2].detach().cpu().numpy()
p_test_exact = uvwp_test_exact[:, :, 3].detach().cpu().numpy()

print("v_test_exact, v_test_pred", v_test_exact.shape, v_test_pred.shape)

l2_error_u = np.linalg.norm(u_test_exact - u_test_pred) / np.linalg.norm(u_test_exact)
l2_error_v = np.linalg.norm(v_test_exact - v_test_pred) / np.linalg.norm(v_test_exact)
l2_error_w = np.linalg.norm(w_test_exact - w_test_pred) / np.linalg.norm(w_test_exact)
l2_error_p = np.linalg.norm(p_test_exact - p_test_pred) / np.linalg.norm(p_test_exact)

print("L2 error velocity u:", l2_error_u)
print("L2 error velocity v:", l2_error_v)
print("L2 error velocity w:", l2_error_w)
print("L2 error pressure p:", l2_error_p)

print("\nEND OF TESTING\n")
#exit()

#np.savetxt('u_test_pred.csv', uvwp_test_pred[:,:,0]., delimiter=',')
#np.savetxt('uvwp_test_exact.csv', uvwp_test_exact, delimiter=',')



