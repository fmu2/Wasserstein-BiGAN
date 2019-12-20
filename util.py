import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd


def log_odds(p):
  p = torch.clamp(p.mean(dim=0), 1e-7, 1-1e-7)
  return torch.log(p / (1 - p))


class MaxOut(nn.Module):
  def __init__(self, k=2):
    """ MaxOut nonlinearity.
    
    Args:
      k: Number of linear pieces in the MaxOut opeartion. Default: 2
    """
    super().__init__()

    self.k = k

  def forward(self, input):
    output_dim = input.size(1) // self.k
    input = input.view(input.size(0), output_dim, self.k, input.size(2), input.size(3))
    output, _ = input.max(dim=2)
    return output


class DeterministicConditional(nn.Module):
  def __init__(self, mapping, shift=None):
    """ A deterministic conditional mapping. Used as an encoder or a generator.

    Args:
      mapping: An nn.Sequential module that maps the input to the output deterministically.
      shift: A pixel-wise shift added to the output of mapping. Default: None
    """
    super().__init__()

    self.mapping = mapping
    self.shift = shift

  def set_shift(self, value):
    if self.shift is None:
      return
    assert list(self.shift.data.size()) == list(value.size())
    self.shift.data = value

  def forward(self, input):
    output = self.mapping(input)
    if self.shift is not None:
      output = output + self.shift
    return output


class GaussianConditional(nn.Module):
  def __init__(self, mapping, shift=None):
    """ A Gaussian conditional mapping. Used as an encoder or a generator.

    Args:
      mapping: An nn.Sequential module that maps the input to the parameters of the Gaussian.
      shift: A pixel-wise shift added to the output of mapping. Default: None
    """
    super().__init__()

    self.mapping = mapping
    self.shift = shift

  def set_shift(self, value):
    if self.shift is None:
      return
    assert list(self.shift.data.size()) == list(value.size())
    self.shift.data = value

  def forward(self, input):
    params = self.mapping(input)
    nlatent = params.size(1) // 2
    mu, log_sigma = params[:, :nlatent], params[:, nlatent:]
    sigma = log_sigma.exp()
    eps = torch.randn(mu.size()).to(input.device)
    output = mu + sigma * eps
    if self.shift is not None:
      output = output + self.shift
    return output


class JointCritic(nn.Module):
  def __init__(self, x_mapping, z_mapping, joint_mapping):
    """ A joint Wasserstein critic function.

    Args:
      x_mapping: An nn.Sequential module that processes x.
      z_mapping: An nn.Sequential module that processes z.
      joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.
    """
    super().__init__()

    self.x_net = x_mapping
    self.z_net = z_mapping
    self.joint_net = joint_mapping

  def forward(self, x, z):
    assert x.size(0) == z.size(0)
    x_out = self.x_net(x)
    z_out = self.z_net(z)
    joint_input = torch.cat((x_out, z_out), dim=1)
    output = self.joint_net(joint_input)
    return output


class WALI(nn.Module):
  def __init__(self, E, G, C):
    """ Adversarially learned inference (a.k.a. bi-directional GAN) with Wasserstein critic.

    Args:
      E: Encoder p(z|x).
      G: Generator p(x|z).
      C: Wasserstein critic function f(x, z).
    """
    super().__init__()

    self.E = E
    self.G = G
    self.C = C

  def get_encoder_parameters(self):
    return self.E.parameters()

  def get_generator_parameters(self):
    return self.G.parameters()

  def get_critic_parameters(self):
    return self.C.parameters()

  def encode(self, x):
    return self.E(x)

  def generate(self, z):
    return self.G(z)

  def reconstruct(self, x):
    return self.generate(self.encode(x))

  def criticize(self, x, z_hat, x_tilde, z):
    input_x = torch.cat((x, x_tilde), dim=0)
    input_z = torch.cat((z_hat, z), dim=0)
    output = self.C(input_x, input_z)
    data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
    return data_preds, sample_preds

  def calculate_grad_penalty(self, x, z_hat, x_tilde, z):
    bsize = x.size(0)
    eps = torch.rand(bsize, 1, 1, 1).to(x.device) # eps ~ Unif[0, 1]
    intp_x = eps * x + (1 - eps) * x_tilde
    intp_z = eps * z_hat + (1 - eps) * z
    intp_x.requires_grad = True
    intp_z.requires_grad = True
    C_intp_loss = self.C(intp_x, intp_z).sum()
    grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
    grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
    grads = torch.cat((grads_x, grads_z), dim=1)
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

  def forward(self, x, z, lamb=10):
    z_hat, x_tilde = self.encode(x), self.generate(z)
    data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
    EG_loss = torch.mean(data_preds - sample_preds)
    C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
    return C_loss, EG_loss