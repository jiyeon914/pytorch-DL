import sys
import os
import math
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

import torch
from torch import nn
from torch import optim

from data_loader import *
from models import Generator, Discriminator



import torch
from data_loader import read_data, get_batch
from models import Generator

def test(args):
    data_train, data_val, data_test, scaling_factor = read_data(args)
    baseline_CNN = Generator().to(args.device)
    baseline_CNN.load_state_dict(torch.load(args.checkpoint_path))
    baseline_CNN.eval()

    test_dataloader = get_batch(data_test, args, shuffle=False)
    for iter, (X, Y) in enumerate(test_dataloader):
        X, Y = X.to(args.device), Y.to(args.device)
        with torch.no_grad():
            output = baseline_CNN(X)
            # Evaluate the model's performance...

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.Filename = '/data/jykim3994/TurbGAN_torch'
    args.Data_Dir = '/data/jykim3994/1.PANN/Data/resolution 128'
    args.data_res = 128
    args.seq_length = 100
    args.lead_time = 20
    args.start = 150
    args.batch_size = 32
    args.device = "cuda"
    args.checkpoint_path = "/data/jykim3994/TurbGAN_torch/MODELS/20240125 baseline CNN 0.5T_L/50ckpt.pt"
    test(args)




import torch
import hydra
from omegaconf import DictConfig
from data_loader import read_data, get_batch
from models import Generator

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):
    data_train, data_val, data_test, scaling_factor = read_data(cfg)
    baseline_CNN = Generator().to(cfg.device)
    baseline_CNN.load_state_dict(torch.load(cfg.checkpoint_path))
    baseline_CNN.eval()

    test_dataloader = get_batch(data_test, cfg, shuffle=False)
    for iter, (X, Y) in enumerate(test_dataloader):
        X, Y = X.to(cfg.device), Y.to(cfg.device)
        with torch.no_grad():
            output = baseline_CNN(X)
            # Evaluate the model's performance...

if __name__ == "__main__":
    test()




pi = math.pi; exp = torch.exp; nu = 10**-3
nx = 128; ny = 128; mx = 3*nx//2; my = 3*ny//2; a = 0; b = 2*pi; L = b-a
dx = L/nx; dy = L/ny; T = 3*10**1; dt = 2.5*10**-3; n = int(T/dt); K = 20

trans = torch.linspace(a, b, nx+1, dtype = torch.float64)
longi = torch.linspace(a, b, ny+1, dtype = torch.float64)

kx = [(2*pi/L)*px for px in range(nx//2+1)]
kx = torch.unsqueeze(torch.DoubleTensor(kx), dim = 0)
ky = [(2*pi/L)*py if py < ny/2 else (2*pi/L)*py-ny for py in range(ny)]
ky = torch.unsqueeze(torch.DoubleTensor(ky), dim = -1)

k_mag = kx**2+ky**2
k_round = torch.round(np.sqrt(k_mag)).to(torch.int16); k_max = torch.max(k_round)
k_index, k_count = torch.unique(k_round, return_counts = True)

def setup_logging(Filename, model_case):
    os.makedirs(os.path.join(Filename, "MODELS", model_case), exist_ok = True)
    os.makedirs(os.path.join(Filename, model_case), exist_ok = True)



import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args(args = [])
args.model_case = '20240125 baseline CNN 0.5T_L'
# args.Filename = '/data/jykim3994/TurbGAN_torch'
# args.Data_Dir = '/data/jykim3994/1.PANN/Data/resolution 128'
args.Filename = '/home/jykim3994/1.PANN'
args.Data_Dir = args. Filename + '/Data/resolution 128'
args.data_res = 128
args.seq_length = 100
args.lead_time = 20
args.start = 150
args.epochs = 200
args.batch_size = 32
args.lr = 1e-4
args.device = "cuda"

setup_logging(args.Filename, args.model_case)
device = args.device
nt = args.lead_time; start = args.start; use_data = args.seq_length + nt + 1

data_train, _, data_test, scaling_factor = read_data(args)
num_train = data_train.shape[0]; num_test = data_test.shape[0]
args.batch_size = num_test #10
test_loader = get_batch(data_test, args, shuffle = False); print(len(test_loader))

baseline_CNN = Generator().to(device)
CNN_Restore = os.path.join(args.Filename, "MODELS", args.model_case, f"{args.epochs}ckpt.pt")
baseline_CNN.load_state_dict(torch.load(CNN_Restore))



# args.run_name = args.model_case
target = data_test[:,nt:,:,:,:].to(torch.float64)
pred = torch.zeros((num_test*(use_data-nt),1,ny,nx), dtype = torch.float64)
with torch.inference_mode():
    for q, (X, Y) in enumerate(test_loader):
        X = X.to(device); Y = Y.to(device)
        logging.info(f"Prediction results of test data t{q} input")

        y_pred = baseline_CNN(X)
        pred[q*num_test:(q+1)*num_test,:,:,:] = y_pred.to(device = 'cpu', dtype = torch.float64)
        logging.info(f"batch {q+1} done.")
pred = rearrange(pred, "(t s) c h w -> s t c h w", s = num_test)

# for num in range(num_test):
#     if num + 1 != 1 and num + 1 != num_test: continue
#     Filewrite = os.path.join(args.Filename, args.model_case, f"{args.run_name} test{num+1} prediction.plt")
#     fvis = open(Filewrite, 'w')
#     fvis.write('VARIABLES="x/pi","y/pi","w"\n')
#     for q in range(use_data-nt):
#         fvis.write(f'Zone T="T=%.4lf" I={nx} J={ny}\n' %(dt*K*(q+start+nt)))
#         for j in range(ny):
#             for i in range(nx):
#                 fvis.write('%lf %lf %lf\n' %(trans[i]/pi,longi[j]/pi,pred[num,q,0,j,i]))
# fvis.close()



target = target[:,:,0,:,:]
tar_mean = torch.mean(target, dim = (2,3), keepdims = True)
tar_fluc = target - tar_mean
tar_rms = torch.std(tar_fluc, dim = (2,3), correction = 0)
tar_skew = torch.mean(tar_fluc**3, dim = (2,3))/tar_rms**3
tar_flat = torch.mean(tar_fluc**4, dim = (2,3))/tar_rms**4

pred = pred[:,:,0,:,:]
pred_mean = torch.mean(pred, dim = (2,3), keepdims = True)
pred_fluc = pred - pred_mean
pred_rms = torch.std(pred_fluc, dim = (2,3), correction = 0)
pred_skew = torch.mean(pred_fluc**3, dim = (2,3))/pred_rms**3
pred_flat = torch.mean(pred_fluc**4, dim = (2,3))/pred_rms**4

corr_coeff = torch.mean(tar_fluc*pred_fluc, dim = (2,3))/(tar_rms*pred_rms)
error = pred - target
MSE = torch.mean(error**2, dim = (2,3))

corr_coeff_avg = torch.mean(corr_coeff, dim = 0)
MSE_avg = torch.mean(MSE, dim = 0)
logging.info(f"Test result t0 input, CC={corr_coeff_avg[0]} MSE={MSE_avg[0]}")

tar_rms_avg = torch.mean(tar_rms, dim = 0)
tar_skew_avg = torch.mean(tar_skew, dim = 0)
tar_flat_avg = torch.mean(tar_flat, dim = 0)
pred_rms_avg = torch.mean(pred_rms, dim = 0)
pred_skew_avg = torch.mean(pred_skew, dim = 0)
pred_flat_avg = torch.mean(pred_flat, dim = 0)

args.run_name = '20240125' + args.model_case[8:]
Filewrite = os.path.join(args.Filename, args.model_case, f"{args.run_name} avg CC.plt")
fw = open(Filewrite, 'w')
fw.write('VARIABLES="t","Corr_Coef"\n')
fw.write('ZONE T="correlation coefficient"\n')
for q in range(use_data-nt):
    fw.write('%.2lf %.10lf\n'%(dt*K*(q+start+nt),corr_coeff_avg[q]))
fw.close()

Filewrite = os.path.join(args.Filename, args.model_case, f"{args.run_name} avg final MSE.plt")
fw = open(Filewrite, 'w')
fw.write('VARIABLES="t","MSE"\n')
fw.write('ZONE T="Mean Square Error"\n')
for q in range(use_data-nt):
    fw.write('%.2lf %.10lf\n'%(dt*K*(q+start+nt),MSE_avg[q]))
fw.close()

Filewrite = os.path.join(args.Filename, args.model_case, f"{args.run_name} avg statistics.plt")
fw = open(Filewrite, 'w')
fw.write('VARIABLES="t","target_rms","target_skew","target_flat","pred_rms","pred_skew","pred_flat"\n')
fw.write('ZONE T="Statistics" I=%d\n' %(use_data-nt))
for q in range(use_data-nt):
    fw.write('%.2lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n' %(dt*K*(q+start+nt),tar_rms_avg[q],tar_skew_avg[q],\
                                                                tar_flat_avg[q],pred_rms_avg[q],pred_skew_avg[q],pred_flat_avg[q]))
fw.close()



tark = torch.fft.rfft2(target, dim = (2,3), norm = 'forward')
predk = torch.fft.rfft2(pred, dim = (2,3), norm = 'forward')

tar_ens_2D = pi*torch.reshape(k_round, [1,1,ny,nx//2+1])*torch.abs(tark)**2
pred_ens_2D = pi*torch.reshape(k_round, [1,1,ny,nx//2+1])*torch.abs(predk)**2

tar_ens_spectrum = torch.zeros([num_test,use_data-nt,k_max+1], dtype = torch.float64)
pred_ens_spectrum = torch.zeros([num_test,use_data-nt,k_max+1], dtype = torch.float64)
for j in range(ny):
    for i in range(nx//2+1):
        tar_ens_spectrum[:,:,k_round[j,i]] += tar_ens_2D[:,:,j,i]
        pred_ens_spectrum[:,:,k_round[j,i]] += pred_ens_2D[:,:,j,i]
tar_ens_spectrum_avg = torch.mean(tar_ens_spectrum, dim = 0)
pred_ens_spectrum_avg = torch.mean(pred_ens_spectrum, dim = 0)

Filewrite1 = os.path.join(args.Filename, args.model_case, f"{args.run_name} avg enstrophy spectrum.plt")
fE = open(Filewrite1, 'w')
fE.write('VARIABLES="k","Target","Prediction"\n')
for q in range(use_data-nt):
    fE.write('ZONE T="T=%.4lf"\n' %(dt*K*(q+start+nt)))
    for i in range(k_max+1):
        fE.write('%d %.12lf %.12lf\n' %(k_index[i],tar_ens_spectrum_avg[q,i],pred_ens_spectrum_avg[q,i]))
fE.close()
