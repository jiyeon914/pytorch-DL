import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
import math
pi = math.pi
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S")

import torch
exp = torch.exp
from torch import nn, optim
import hydra
from omegaconf import DictConfig

from dataloader.data_loader import read_data, get_batch
from models.model import BaselineCNN
from models.loss import CNN_Loss
from utils.util import setup_logging, init_file



nu = 10**-3
nx = 128; ny = 128; mx = 3*nx//2; my = 3*ny//2; a = 0; b = 2*pi; L = b-a
dx = L/nx; dy = L/ny; T = 3*10**1; dt = 2.5*10**-3; n = int(T/dt); K = 20

trans = torch.linspace(a, b, nx+1, dtype = torch.float64)
longi = torch.linspace(a, b, ny+1, dtype = torch.float64)

kx = [(2*pi/L)*px for px in range(nx//2+1)]
kx = torch.unsqueeze(torch.DoubleTensor(kx), dim = 0)
ky = [(2*pi/L)*py if py < ny/2 else (2*pi/L)*py-ny for py in range(ny)]
ky = torch.unsqueeze(torch.DoubleTensor(ky), dim = -1)

k_mag = kx**2+ky**2
k_round = torch.round(torch.sqrt(k_mag)).to(torch.int16); k_max = torch.max(k_round)
k_index, k_count = torch.unique(k_round, return_counts = True)



@hydra.main(version_base=None, config_path="../config", config_name="default_config")
def train(cfg: DictConfig):
    device = cfg.device; batch_size = cfg.batch_size
    nt = cfg.nt; start = cfg.start; use_data = cfg.seq_length + nt + 1
    
    setup_logging(cfg)
    data_train, data_val, data_test, _ = read_data(cfg)
    baseline_CNN = BaselineCNN().to(device)
    optimizer_CNN = optim.Adam(baseline_CNN.parameters(), lr=cfg.lr, betas=(0, 0.999))
    mse = nn.MSELoss()
    huber = nn.HuberLoss()

    file_path = {
        "total_loss": os.path.join(cfg.file_dir, cfg.run_name, f"{cfg.run_name} total loss.plt"),
        "data_loss": os.path.join(cfg.file_dir, cfg.run_name, f"{cfg.run_name} training MSE.plt"),
        "reg_loss": os.path.join(cfg.file_dir, cfg.run_name, f"{cfg.run_name} regularization loss.plt"),
    }
    init_file(file_path['total_loss'], '"epoch","Training loss","Validation loss"', '"Loss"')
    init_file(file_path['data_loss'], '"epoch","Training MSE","Validation MSE"', '"MSE"')
    init_file(file_path['reg_loss'], '"epoch","Training reg loss","Validation reg loss"', '"Reg loss"')

    train_dataloader = get_batch(data_train, cfg, shuffle=True)
    for _, (X, Y) in enumerate(train_dataloader):
        train_err_X = X.to(device); train_err_Y = Y.to(device)
        break
    val_dataloader = get_batch(data_val, cfg, shuffle=True)
    for _, (X, Y) in enumerate(val_dataloader):
        val_err_X = X.to(device); val_err_Y = Y.to(device)
        break

    for epoch in range(1, cfg.epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        train_dataloader = get_batch(data_train, cfg, shuffle=True); l = len(train_dataloader)
        for iter, (X, Y) in enumerate(train_dataloader):
            # spatial shift
            del_x = 2*pi*torch.rand((1,)); del_y = 2*pi*torch.rand((1,))
            k_dot_delta = torch.reshape(del_x*kx + del_y*ky, (1,1,ny,nx//2+1)).to(torch.float32)
            Xk = torch.fft.rfft2(X, dim = (2,3), norm = 'forward'); Yk = torch.fft.rfft2(Y, dim = (2,3), norm = 'forward')
            Xk_shift = exp(-1J*k_dot_delta)*Xk; Yk_shift = exp(-1J*k_dot_delta)*Yk
            X = torch.fft.irfft2(Xk_shift, dim = (2,3), norm = 'forward'); Y = torch.fft.irfft2(Yk_shift, dim = (2,3), norm = 'forward')
            X = X.to(device); Y = Y.to(device)

            optimizer_CNN.zero_grad()
            loss_CNN, data_loss, reg_loss = CNN_Loss(baseline_CNN, X, Y, mse)
            loss_CNN.backward()
            optimizer_CNN.step()

            if iter % 100 == 0:
                logging.info(f"{loss_CNN.item()}, {data_loss.item()}, {reg_loss.item()}")

            if epoch == 1 and iter == 0:
                baseline_CNN.eval()
                MSE_curr = 0
                for _, (X_err, Y_err) in enumerate(train_dataloader):
                    X_err = X_err.to(device); Y_err = Y_err.to(device)
                    mse_curr_pred = baseline_CNN(X_err)
                    MSE_curr += mse(mse_curr_pred, Y_err).item()
                MSE_curr /= l

                MSE_val_curr = 0
                for _, (X_err, Y_err) in enumerate(val_dataloader):
                    X_err = X_err.to(device); Y_err = Y_err.to(device)
                    mse_curr_pred = baseline_CNN(X_err)
                    MSE_val_curr += mse(mse_curr_pred, Y_err).item()
                MSE_val_curr /= len(val_dataloader)

                total_curr, _, reg_curr = CNN_Loss(baseline_CNN, train_err_X, train_err_Y, mse)
                total_curr, reg_curr = total_curr.item(), reg_curr.item()

                total_val, _, reg_val = CNN_Loss(baseline_CNN, val_err_X, val_err_Y, mse)
                total_val, reg_val = total_val.item(), reg_val.item()

                baseline_CNN.train()
                logging.info(f"End of epoch {epoch-1} total loss {total_curr}, MSE {MSE_curr}, MSE val {MSE_val_curr}, regularization {reg_curr}")

                fw = open(file_path['total_loss'], 'a')
                fw.write('%d %.10f %.10f \n' %(epoch-1, total_curr, total_val))
                fw.close()

                fMSE = open(file_path['data_loss'], 'a')
                fMSE.write('%d %.10f %.10f \n' %(epoch-1, MSE_curr, MSE_val_curr))
                fMSE.close()

                fr = open(file_path['reg_loss'], 'a')
                fr.write('%d %.10f %.10f \n' %(epoch-1, reg_curr, reg_val))
                fr.close()
            else:
                continue

        baseline_CNN.eval()
        MSE_curr = 0
        for _, (X_err, Y_err) in enumerate(train_dataloader):
            X_err = X_err.to(device); Y_err = Y_err.to(device)
            mse_curr_pred = baseline_CNN(X_err)
            MSE_curr += mse(mse_curr_pred, Y_err).item()
        MSE_curr /= l

        MSE_val_curr = 0
        for _, (X_err, Y_err) in enumerate(val_dataloader):
            X_err = X_err.to(device); Y_err = Y_err.to(device)
            mse_curr_pred = baseline_CNN(X_err)
            MSE_val_curr += mse(mse_curr_pred, Y_err).item()
        MSE_val_curr /= len(val_dataloader)

        total_curr, _, reg_curr = CNN_Loss(baseline_CNN, train_err_X, train_err_Y, mse)
        total_curr, reg_curr = total_curr.item(), reg_curr.item()

        total_val, _, reg_val = CNN_Loss(baseline_CNN, val_err_X, val_err_Y, mse)
        total_val, reg_val = total_val.item(), reg_val.item()

        baseline_CNN.train()
        logging.info(f"End of epoch {epoch} total loss {total_curr}, MSE {MSE_curr}, MSE val {MSE_val_curr}, regularization {reg_curr}")

        fw = open(file_path['total_loss'], 'a')
        fw.write('%d %.10f %.10f \n' %(epoch, total_curr, total_val))
        fw.close()

        fMSE = open(file_path['data_loss'], 'a')
        fMSE.write('%d %.10f %.10f \n' %(epoch, MSE_curr, MSE_val_curr))
        fMSE.close()

        fr = open(file_path['reg_loss'], 'a')
        fr.write('%d %.10f %.10f \n' %(epoch, reg_curr, reg_val))
        fr.close()

        if epoch % 50 == 0:
            torch.save(baseline_CNN.state_dict(), os.path.join(cfg.file_dir, "MODELS", cfg.run_name, f"{epoch}ckpt.pt"))

            test_dataloader = get_batch(data_test[-1:,:,:,:,:], cfg, shuffle=False)
            for it, (X, Y) in enumerate(test_dataloader):
                test_X = X.to(device)
                if it == 0:
                    pred = baseline_CNN(test_X).to(device='cpu', dtype=torch.float64)
                else:
                    pred = torch.cat((pred,baseline_CNN(test_X).to(device='cpu', dtype=torch.float64)), dim=0)

            Filewrite = os.path.join(cfg.file_dir, cfg.run_name, f"{cfg.run_name} test50 prediction epoch={epoch}.plt")
            fvis = open(Filewrite, 'w')
            fvis.write('VARIABLES="x/pi","y/pi","w"\n')
            for q in range(use_data-nt):
                fvis.write('ZONE T="T=%.4lf" I=%d J=%d\n' %(dt*K*(q+start+nt),nx,ny))
                for j in range(ny):
                    for i in range(nx):
                        fvis.write('%lf %lf %.10lf\n'%(trans[i]/pi,longi[j]/pi,pred[q,0,j,i]))
            fvis.close()


            target = data_test[-1,nt:,:,:,:]
            tark = torch.fft.rfft2(target, dim=(2,3), norm='forward')
            predk = torch.fft.rfft2(pred, dim=(2,3), norm='forward')

            tar_ens_2D = pi*torch.reshape(k_round, [1,1,ny,nx//2+1])*torch.abs(tark)**2
            pred_ens_2D = pi*torch.reshape(k_round, [1,1,ny,nx//2+1])*torch.abs(predk)**2

            tar_ens_spectrum = torch.zeros([use_data-nt,k_max+1], dtype=torch.float64)
            pred_ens_spectrum = torch.zeros([use_data-nt,k_max+1], dtype=torch.float64)
            for j in range(ny):
                for i in range(nx//2+1):
                    tar_ens_spectrum[:,k_round[j,i]] += tar_ens_2D[:,0,j,i]
                    pred_ens_spectrum[:,k_round[j,i]] += pred_ens_2D[:,0,j,i]

            Filewrite = os.path.join(cfg.file_dir, cfg.run_name, f"{cfg.run_name} test50 Enstrophy epoch={epoch}.plt")
            fE = open(Filewrite, 'w')
            fE.write('VARIABLES="k","Target","Prediction"\n')
            for q in range(use_data-nt):
                fE.write('ZONE T="T=%.4lf"\n' %(dt*K*(q+start+nt)))
                for i in range(k_max+1):
                    fE.write('%d %.10lf %.10lf\n' %(k_index[i],tar_ens_spectrum[q,i],pred_ens_spectrum[q,i]))
            fE.close()   

if __name__ == '__main__':
    train()
