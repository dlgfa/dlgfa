"""Bayesian group lasso in a VAE setup on the bars data.

* Point estimates on the sparse weight parameters.
* Proximal gradient steps are taken to handle the group lasso penalty.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

from lib.bars_data import sample_one_bar_image
                           
from lib.distributions import Normal
from lib.models import BayesianGroupLassoGenerator, NormalNet
from lib.dlgfa import NormalPriorTheta, dlgfa
from lib.utils import Lambda

torch.manual_seed(0)

image_size = 8
dim_z = image_size
dim_x = image_size * image_size
dim_h = 100
n_layers = 1
num_groups = image_size
group_input_dim = 1
seq_len = 20
prior_theta_scale = 1
lam = 5
lam_adjustment = 1

num_samples = 2000
num_train_samples = 1600
num_test_samples = 200
num_val_samples = 200
num_epochs = 10000
mc_samples = 1
batch_size = 32 


# One bar per image
X = torch.stack([
  sample_one_bar_image(image_size).view(-1)
  for _ in range(num_samples)
])

X += 0.05 * torch.randn(X.size())
X = torch.stack([X for _ in range(seq_len)])


stddev_multiple = 0.1


inference_net = NormalNet(
  mu_net= nn.Sequential(
   
    nn.Linear(dim_h + dim_h, dim_z)),
 
  sigma_net=torch.nn.Sequential(
   
    nn.Linear(dim_h + dim_h, dim_z),
    Lambda(torch.exp),
    Lambda(lambda x: x * stddev_multiple + 1e-3)
  )
)

def make_group_generator():
  # Note that this Variable is NOT going to show up in `net.parameters()` and
  # therefore it is implicitly free from the ridge penalty/p(theta) prior.
  log_sigma = Variable(
    torch.log(1e-2 * torch.ones(image_size)),
    requires_grad=True
  )
  return NormalNet(
    mu_net=torch.nn.Sequential(
   
      torch.nn.Linear(group_input_dim,image_size)
    ),
    sigma_net=Lambda(
      lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1)) + 1e-3,
      extra_args=(log_sigma,)
    )
  )

generative_net = BayesianGroupLassoGenerator(
  seq_len=seq_len,
  group_generators=[make_group_generator() for _ in range(image_size)],
  group_input_dim=group_input_dim,
  dim_z=dim_z,
  dim_h=dim_h
)

def debug(count):
  """Create a plot showing the first `count` training samples along with their
  mean z value, x mean, x standard deviation, and a sample from the full model
  (sample z and then sample x). here we plot the true/generated image from time
  point t=8"""
  fig, ax = plt.subplots(5, count, figsize=(12, 4))

  # True images
  for i in range(count):
    ax[0, i].imshow(Xvar[8][i].data.cpu().view(image_size, image_size).numpy())
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])

  # latent representation
  for i in range(count):
    #Xvar = Variable(X)
    #ax[1, i].bar(range(dim_z), inference_net(Xvar).mu.data.squeeze().numpy())
    ax[1,i].bar(range(dim_z), info["all_enc_mean"][8][i].data.squeeze().numpy())
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  # Reconstructed images
  for i in range(count):
    
    fX = info["all_dec_mean"][8][i].view(image_size,image_size)
    
    ax[2, i].imshow(fX.data.squeeze().numpy())
    ax[2, i].axes.xaxis.set_ticks([])
    ax[2, i].axes.yaxis.set_ticks([])

  for i in range(count):
    
    fX = info["all_dec_std"][8][i].view(image_size,image_size)
    ax[3, i].imshow(fX.data.squeeze().numpy())
    ax[3, i].axes.xaxis.set_ticks([])
    ax[3, i].axes.yaxis.set_ticks([])

  for i in range(count):
    
    fX = (info["all_gr2"][8][i]).view(image_size, image_size)
    
    ax[4, i].imshow(fX.data.squeeze().numpy())
    ax[4, i].axes.xaxis.set_ticks([])
    ax[4, i].axes.yaxis.set_ticks([])

  ax[0, 0].set_ylabel('true image')
  ax[1, 0].set_ylabel('z')
  ax[2, 0].set_ylabel('x mu')
  ax[3, 0].set_ylabel('x sigma')
  ax[4, 0].set_ylabel('x sample')

  return fig

def debug_incoming_weights():
  fig, ax = plt.subplots(1, image_size, figsize=(12, 4))

  cmap = 'bwr'
  for i in range(generative_net.Ws.size(1)):
    m = generative_net.Ws[8][i]
    ax[i].imshow(torch.stack([m.data for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
    ax[i].set_title('group {}'.format(i))
    ax[i].set_xlabel('z_i')
    ax[i].axes.xaxis.set_ticks(range(dim_z))
    ax[i].axes.yaxis.set_ticks([])

  ax[0].set_ylabel('learned weights')
  

  return fig

def debug_outgoing_weights():
  fig, ax = plt.subplots(1, dim_z, figsize=(12, 4))

  # rows correspond to groups and cols correspond to z_i's
  col_norms = torch.stack([
    torch.sqrt(torch.sum(torch.pow(generative_net.Ws[8][i].data.t(), 2), dim=0))
    for i in range(generative_net.Ws.size(1))
  ])

  # See https://matplotlib.org/examples/color/colormaps_reference.html
  cmap = 'bwr'
  for i in range(dim_z):
    ax[i].imshow(torch.stack([col_norms[:, i] for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
    ax[i].set_title('z_{}'.format(i))
    ax[i].set_xlabel('groups')
    ax[i].axes.xaxis.set_ticks(range(image_size))
    ax[i].axes.yaxis.set_ticks([])

  return fig

def debug_z_by_group_matrix():
  fig, ax = plt.subplots()
  W_col_norms = torch.sqrt(
    torch.sum(torch.pow(generative_net.Ws[8].data, 2), dim=2)
  )
  W_col_norms_prop = W_col_norms / torch.max(W_col_norms, dim=0)[0]
  ax.imshow(W_col_norms_prop.numpy(), aspect='equal')
  ax.set_xlabel('dimensions of z')
  ax.set_ylabel('group generative nets')
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
 

lr = 1e-3
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
#  {'params': [inference_net_log_stddev], 'lr': lr},
  {'params': generative_net.group_generators_parameters(), 'lr': lr},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr}
])



Ws_lr = 1e-4
optimizer_Ws = torch.optim.SGD([
  {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
])

model = dlgfa(
  inference_model=inference_net,
  generative_model=generative_net,
  prior_theta=NormalPriorTheta(prior_theta_scale),
  lam=lam,
  optimizers=[optimizer, optimizer_Ws],
  dim_x = dim_x,
  dim_h = dim_h,
  dim_z = dim_z,
  n_layers = n_layers
)

plot_interval = 1000
elbo_per_iter = []
for i in range(num_epochs):
  if i > 100:
    stddev_multiple = 1
  Xvar = Variable(X[:,torch.randperm(num_train_samples)[:batch_size]])
  info = model.step(
    X=Xvar,
    prox_step_size=Ws_lr * lam * lam_adjustment,
    mc_samples=mc_samples
  )
  elbo_per_iter.append(info['elbo'].data[0])

  if i % plot_interval == 0 and i > 0:
    debug(8)
    plt.suptitle('DLGFA, Iteration {}, lr = {}, lam = {}, batch_size = {}, num_train_samples = {}'.format(i, lr, lam, batch_size, num_train_samples))

    debug_incoming_weights()
    plt.suptitle('incoming z weights')

    debug_outgoing_weights()
    plt.suptitle('outgoing z weight norms')

    debug_z_by_group_matrix()

    plt.figure()
    plt.plot(elbo_per_iter)
    plt.xlabel('iteration')
    plt.ylabel('ELBO')
    plt.title('ELBO per iteration. lam = {}'.format(lam))
    plt.show()

  # Print the learned (but fixed) standard deviations of each of the generators
  # print(torch.exp(torch.stack([gen.sigma_net.extra_args[0] for gen in generative_net.group_generators])) + 1e-3)

  print('iter', i)
  print('  ELBO:', info['elbo'].data[0])
  print('    -KL(q(z) || p(z))', -info['z_kl'].data[0])
  print('    loglik_term      ', info['loglik_term'].data[0])
  print('    log p(theta)     ', info['logprob_theta'].data[0])
  print('    log p(W)         ', info['logprob_W'].data[0])

# Plot the final connectivity matrix and save
debug_z_by_group_matrix()
plt.savefig('bars_data_connectivity_matrix.pdf', format='pdf')

  
  
def save_img_and_reconstruction(ix):  


  plt.figure()
  plt.imshow(Xvar[8][ix].data.cpu().view(image_size, image_size).numpy())
  plt.savefig('{}_true.pdf'.format(ix), format='pdf')
  fX = (info["all_gr2"][8][i]).view(image_size, image_size)
  plt.imshow(fX.data.squeeze().numpy())
  plt.savefig('{}_reconstruction_full_sample.pdf'.format(ix), format='pdf')
 
    


import statistics
X_val = X[:,(num_train_samples + num_test_samples):num_samples]
Xvar_val = Variable(X_val)
info_val = model.step(
    X=Xvar_val,
    prox_step_size=Ws_lr * lam * lam_adjustment,
    mc_samples=mc_samples
  )


def mse(t):
  
    n = Xvar_val.size(1)
    m = Xvar_val.size(2)
    MSE = []
    for i in range(0,n):
        summation = 0
        for j in range(0,m):
    
            diff = X_val[t][i][j] - info_val["all_gr2"][t][i][j].data.numpy()[-1]
            squared_diff = diff**2
            summation = summation + squared_diff
        MSE.append(summation/m)
    
    Mean_mse = statistics.mean(MSE)
    STD_mse =  statistics.stdev(MSE)
    
    loglike = info["loglik_term"].data.numpy()[-1]
            
    print("MSE, STD and loglikelihood are", Mean_mse, STD_mse,loglike)

def density(t):
    import seaborn as sns
    import math
    n = info["all_dec_mean"][t].size(0)
    m = info["all_dec_mean"][t].size(1)
    mean = []
    var = []
    for i in range(0,n):
        for j in range(0,m):
            mean.append(info["all_dec_mean"][t][i][j].data.numpy()[-1])
            var.append(math.log10(info["all_dec_std"][t][i][j].data.numpy()[-1]))
    #for item in methods:
    sns.distplot(mean,hist=False, kde=True,
                 kde_kws = {'linewidth':2},
                 label = "DLGFA")
    
    
    
    plt.legend(prop={"size":16}, title = "Methods")
    plt.title("Density plot of generated mean")
    #plt.xlabel(")
    plt.ylabel("Density")
        
        
    
    
