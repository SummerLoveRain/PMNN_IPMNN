import logging
import time
import numpy as np
import torch
from base_config import BaseConfig


# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)


class PINNConfig(BaseConfig):
    def __init__(self, param_dict, train_dict, model):
        super().__init__()
        self.init()
        self.model = model
        # 设置使用设备:cpu, cuda
        lb, ub, self.device, self.path, self.root_path = self.unzip_param_dict(
            param_dict=param_dict)
        # 上下界
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        # 加载训练参数
        self.lambda_true, x, self.d, self.N_R = self.unzip_train_dict(
            train_dict=train_dict)

        # 区域内点
        self.x = []
        for i in range(self.d):
            xi = x[:, i:i+1]
            X = self.data_loader(xi)
            self.x.append(X)

        # 初始猜测
        # u = np.zeros(shape=[self.N_R, 1])
        # u[0, 0] = 1
        u = np.random.rand(self.N_R, 1)
        u = u/np.linalg.norm(u)
        self.u = self.data_loader(u, requires_grad=False)
        self.lambda_last = 1

        self.lambda_ = None

    # 训练参数初始化
    def init(self, loss_name='mean', model_name='PINN'):
        self.start_time = None
        # 小于这个数是开始保存模型
        self.min_loss = 1e20
        # 记录运行步数
        self.nIter = 0
        # 损失计算方式
        if loss_name == 'sum':
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        else:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        # 保存模型的名字
        self.model_name = model_name

    # 参数读取

    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['lb'], param_dict['ub'],
                      param_dict['device'], param_dict['path'],
                      param_dict['root_path'])
        return param_data

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['lambda_'],
            train_dict['x'],
            train_dict['d'],
            train_dict['N_R'],
        )
        return train_data

    def net_model(self, x):
        if isinstance(x, list):
            X = torch.cat((x), 1)
        else:
            X = x
        X = self.coor_shift(X, self.lb, self.ub)
        result = self.model.forward(X)
        # 强制Dirichlet边界条件
        g_x = 1
        for i in range(self.d):
            # g_x = g_x * (1-torch.exp(-(x[i]-self.lb[i])))*(1-torch.exp(-(x[i]-self.ub[i])))
            g_x = g_x * (torch.exp((x[i]-self.lb[i]))-1)*(torch.exp(-(x[i]-self.ub[i]))-1)
        result = g_x * result
        return result

    def forward(self, x):
        result = self.net_model(x)
        return result

    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()
        # 区域点
        x = self.x
        u = self.forward(x)
        u_xx = None
        for i in range(self.d):
            xi = x[i]
            u_xi = self.compute_grad(u, xi)
            u_xixi = self.compute_grad(u_xi, xi)
            if u_xx is None:
                u_xx = u_xixi
            else:
                u_xx = u_xx + u_xixi

        # Rayleigh-Quotient 计算最小特征值 lambda_
        # <Lu, u>/<u, u>
        Lu = -u_xx

          # 由于是使用反幂法，则要求 Lu^(k+1) = lambda u^k, 因此新增一个损失
        # loss_IPM = self.loss_func(Lu - self.lambda_last * self.u)   
        loss_IPM = self.loss_func(Lu/torch.norm(Lu) - self.u)
         
        # 权重
        alpha_IPM = 1

        self.loss = loss_IPM*alpha_IPM

        # 反向传播
        self.loss.backward()
        # 运算次数加1
        self.nIter = self.nIter + 1

        # # 区域计算Rayleigh-Quotient
        Luu = torch.sum(Lu*u)
        uu = torch.sum(u**2)
        lambda_ = self.detach(Luu/uu)
        lambda_ = lambda_.max()
        self.lambda_last = lambda_
        # loss = self.loss_func(Lu - lambda_*u)

        # 更新u^k = u^(k+1)
        # 归一化
        norm_u = u/torch.norm(u, p=2)
        # # norm_u = self.detach(norm_u)

        # 保存模型
        # loss = self.detach(self.loss + torch.norm(self.u - norm_u))
        loss = self.detach(self.loss)
        # loss = self.detach(torch.norm(self.u - norm_u))
        self.u = self.data_loader(self.detach(norm_u), requires_grad=False)
        # self.u = self.data_loader(u)
        if loss < self.min_loss:
            self.lambda_ = lambda_
            self.min_loss = loss
            PINNConfig.save(net=self,
                            path=self.root_path + '/' + self.path,
                            name=self.model_name)

        # 打印日志
        loss_remainder = 1
        if np.remainder(self.nIter, loss_remainder) == 0:
            # 打印常规loss
            loss_IPM = self.detach(loss_IPM)

            abs_lambda = np.abs(self.lambda_true-self.lambda_)
            rel_lambda = abs_lambda/self.lambda_true


            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss ' + str(loss) +\
                ' lambda_ ' + str(lambda_) +' loss_IPM ' + str(loss_IPM) + \
                ' min_loss ' + str(self.min_loss) +\
                ' lambda ' + str(self.lambda_) + ' abs_lambda '+str(abs_lambda)+' rel_lambda '+str(rel_lambda)+\
                ' LR ' + str(self.optimizer.state_dict()['param_groups'][0]['lr'])

            log(log_str)

            # 打印耗时
            elapsed = time.time() - self.start_time
            print('Time: %.4fs Per %d Iterators' % (elapsed, loss_remainder))
            logging.info('Time: %.4f s Per %d Iterators' %
                         (elapsed, loss_remainder))
            self.start_time = time.time()
        return self.loss
