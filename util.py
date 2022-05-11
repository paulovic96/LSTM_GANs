import numpy as np
import torch

def b2mb(x): return int(x/2**20)
class TorchTracemalloc():

    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used   = b2mb(self.end-self.begin)
        self.peaked = b2mb(self.peak-self.begin)
        print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def add_and_pad(xx, max_len):
    """
    Padd a sequence with last value to maximal length

    :param xx: 2D np.array
        seuence to be padded (seq_length, feeatures)
    :param max_len: int
        maximal length to be padded to
    :param with_onset_dim: bool
        add one features with 1 for the first time step and rest 0 to indicate sound onset
    :return: 2D torch.Tensor
        padded sequence
    """
    if len(xx.shape)>1:
        seq_length = xx.shape[0]
    else:
        seq_length = 1
    padding_size = max_len - seq_length
    padding_size = tuple([padding_size] + [1 for i in range(len(xx.shape) - 1)])
    xx = np.concatenate((xx, np.tile(xx[-1:], padding_size)), axis=0)
    return torch.from_numpy(xx)


def pad_batch_online(lens, data_to_pad, device = "cpu"):
    """
    :param lens: 1D torch.Tensor
        Tensor containing the length of each sample in data_to_pad of one batch
    :param data_to_pad: series
        series containing the data to pad
    :return padded_data: torch.Tensors
        Tensors containing the padded and stacked to one batch
    """
    max_len = int(max(lens))
    padded_data = torch.stack(list(data_to_pad.apply(lambda x: add_and_pad(x, max_len)))).to(device)

    return padded_data


def get_vel_acc_jerk(trajectory, *, lag=1):
    """returns (velocity, acceleration, jerk) tuple"""
    velocity = (trajectory[:, lag:, :] - trajectory[:, :-lag, :]) / lag
    acc = (velocity[:, 1:, :] - velocity[:, :-1, :]) / 1.0
    jerk = (acc[:, 1:, :] - acc[:, :-1, :]) / 1.0
    return velocity, acc, jerk

def velocity_jerk_loss(pred, loss, *, guiding_factor=None):
    """returns (velocity_loss, jerk_loss) tuple"""
    vel1, acc1, jerk1 = get_vel_acc_jerk(pred)
    vel2, acc2, jerk2 = get_vel_acc_jerk(pred, lag=2)
    vel4, acc4, jerk4 = get_vel_acc_jerk(pred, lag=4)

    loss = rmse_loss

    # in the lag calculation higher lags are already normalised to standard
    # units
    if guiding_factor is None:
        velocity_loss = (loss(vel1, torch.zeros_like(vel1))
                         + loss(vel2, torch.zeros_like(vel2))
                         + loss(vel4, torch.zeros_like(vel4)))
        jerk_loss = (loss(jerk1, torch.zeros_like(jerk1))
                     + loss(jerk2, torch.zeros_like(jerk2))
                     + loss(jerk4, torch.zeros_like(jerk4)))
    else:
        assert 0.0 < guiding_factor < 1.0
        velocity_loss = (loss(vel1, guiding_factor * vel1.detach().clone())
                         + loss(vel2, guiding_factor * vel2.detach().clone())
                         + loss(vel4, guiding_factor * vel4.detach().clone()))
        jerk_loss = (loss(jerk1, guiding_factor * jerk1.detach().clone())
                     + loss(jerk2, guiding_factor * jerk2.detach().clone())
                     + loss(jerk4, guiding_factor * jerk4.detach().clone()))

    return velocity_loss, jerk_loss

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
    
rmse_loss = RMSELoss(eps=0)


def get_decomposition_matrix(cov):
    try:
        return np.linalg.cholesky(cov), "cholesky"
    except np.linalg.LinAlgError as e:
        return np.linalg.svd(cov), "SVD"

def sample_multivariate_normal(mean, decomposition_matrix,decomposition):
    if decomposition == "cholesky":
        standard_normal_vector = np.random.standard_normal(len(decomposition_matrix))
        return decomposition_matrix @ standard_normal_vector + mean
    elif decomposition == "SVD":
        u, s, vh = decomposition_matrix
        standard_normal_vector = np.random.standard_normal(len(u))
        return u @ np.diag(np.sqrt(s)) @ vh @ standard_normal_vector + mean



def cp_trjacetory_loss(Y_hat, tgts):
    """
    Calculate additive loss using the RMSE of position velocity , acc and jerk

    :param Y_hat: 3D torch.Tensor
        model prediction
    :param tgts: 3D torch.Tensor
        target tensor
    :return loss, pos_loss, vel_loss, acc_loss, jerk_loss: torch.Tensor
        summed total loss with all individual losses
    """

    velocity, acc, jerk = get_vel_acc_jerk(tgts)
    velocity2, acc2, jerk2 = get_vel_acc_jerk(tgts, lag=2)
    velocity4, acc4, jerk4 = get_vel_acc_jerk(tgts, lag=4)

    Y_hat_velocity, Y_hat_acceleration, Y_hat_jerk = get_vel_acc_jerk(Y_hat)
    Y_hat_velocity2, Y_hat_acceleration2, Y_hat_jerk2 = get_vel_acc_jerk(Y_hat, lag=2)
    Y_hat_velocity4, Y_hat_acceleration4, Y_hat_jerk4 = get_vel_acc_jerk(Y_hat, lag=4)

    pos_loss = rmse_loss(Y_hat, tgts)
    vel_loss = rmse_loss(Y_hat_velocity, velocity) + rmse_loss(Y_hat_velocity2, velocity2) + rmse_loss(Y_hat_velocity4, velocity4)
    jerk_loss = rmse_loss(Y_hat_jerk, jerk) + rmse_loss(Y_hat_jerk2, jerk2) + rmse_loss(Y_hat_jerk4, jerk4)
    acc_loss = rmse_loss(Y_hat_acceleration, acc) + rmse_loss(Y_hat_acceleration2, acc2) + rmse_loss(Y_hat_acceleration4, acc4)

    
    loss = pos_loss + vel_loss + acc_loss + jerk_loss
    return loss, pos_loss, vel_loss, acc_loss, jerk_loss

