import torch

def CNN_Loss(model, x, y, loss_type):
    y_pred = model(x)
    data_loss = loss_type(y_pred, y)
    l2_reg = sum(0.5*torch.sum(p**2) for _,p in enumerate(model.parameters()) if p.dim() != 1)
    loss_CNN = 1*data_loss + 0.0001*l2_reg
    return loss_CNN, data_loss, l2_reg

def D_Loss(G, D, x, y, args):
    batch_size = x.shape[0]; device = args.device

    y_pred = G(x)
    d_true = D(y, x); d_true_loss = torch.mean(d_true)
    d_false = D(y_pred, x); d_false_loss = torch.mean(d_false)

    epsilon = torch.rand((batch_size, 1, 1, 1), device = device)
    y_hat = (y + epsilon*(y_pred - y)).requires_grad_(True)
    d_hat = D(y_hat, x)
    d_grad = torch.autograd.grad(
                outputs = d_hat,
                inputs = y_hat,
                grad_outputs = torch.ones_like(d_hat, requires_grad = False, device = device),
                create_graph = True,
                retain_graph = True,)[0] # [0]은 왜 필요한 거지?
    # d_grad = d_grad.view(d_grad.size(0), -1)
    slope = torch.sqrt(torch.sum(d_grad**2, dim = (1,2,3))) # dim = (1)이면 1근방에 머물면서 oscillation, dim = (1,2,3)이면 minimize 가능성은 보이나 상당히 크게 요동침
    gp_loss = ((slope - 1.0)**2).mean()

    loss_D = 1*(-d_true_loss + d_false_loss) + 10*gp_loss + 0.001*(d_true_loss**2)
    return loss_D, d_true_loss, d_false_loss, gp_loss

def G_Loss(G, D, x, y, loss_type):
    y_pred = G(x)
    data_loss = loss_type(y_pred, y)
    d_false = D(y_pred, x); d_false_loss = torch.mean(d_false)
    loss_G = 100*data_loss - 1*d_false_loss
    return loss_G, data_loss


