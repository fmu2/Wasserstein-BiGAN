import argparse, torch
from torchvision import utils
from wali_celeba import create_WALI


NLAT = 100
IMAGE_SIZE = 64
NUM_CHANNELS = 3


def interpolate(generator, z0, z1, nintp=10, path='linear', filepath=None):
  """ Interpolate in the latent space.

  Args:
    generator: Generator network that takes z as input.
    z0: Where interpolation starts.
    z1: Where interpolation ends.
    nintp: Number of intermediate steps.
    path: Trajectory of interpolation. Default: linear
    filepath: Where to save the images.
  """
  assert path in ['linear', 'spherical']
  assert z1.size() == z1.size()
  z0, z1 = z0.view(z0.size(0), NLAT, 1, 1), z1.view(z1.size(0), NLAT, 1, 1)
  alphas = torch.linspace(0, 1, nintp)
  imgs = []

  if path == 'linear':
    for alpha in alphas:
      z = z0 * alpha + z1 * (1 - alpha)
      img = generator(z).detach_() * 0.5 + 0.5
      imgs.append(img.cpu())
  elif path == 'spherical':
    nz0, nz1 = z0 / z0.norm(dim=1, keepdim=True), z1 / z1.norm(dim=1, keepdim=True)
    theta = ((nz0 * nz1).sum(dim=1, keepdim=True)).acos()
    for alpha in alphas:
      z = torch.sin((1 - alpha) * theta) / torch.sin(theta) * z0 \
        + torch.sin(alpha * theta) / torch.sin(theta) * z1
      img = generator(z).detach_() * 0.5 + 0.5
      imgs.append(img.cpu())

  imgs = torch.cat(imgs, dim=1).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
  grid = utils.make_grid(imgs, nrow=nintp)
  utils.save_image(grid, filepath)
  print('Interpolated images saved.')


def main(args):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  wali = create_WALI()
  ckpt = torch.load(args.ckpt)
  wali.load_state_dict(ckpt)
  generator = wali.G.to(device)

  z0 = torch.randn(args.n, NLAT, 1, 1).to(device)
  z1 = torch.randn(args.n, NLAT, 1, 1).to(device)
  interpolate(generator, z0, z1, nintp=10, path='linear', filepath=args.save_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Plot interpolations for WALI.')
  parser.add_argument("--ckpt", type=str, help='Path to the saved model checkpoint', default=None)
  parser.add_argument("--n", type=int, help="number of interpolated paths", default=4)
  parser.add_argument("--save-path", type=str, help="where to save the interpolations", default=None)
  args = parser.parse_args()
  main(args)