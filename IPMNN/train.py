import logging
import sys
from turtle import shape
from pyDOE import lhs
from init_config import get_device, path, root_path, init_log, train_Adam, train_Adam_LBFGS, train_Adam_ResNet, train_LBFGS
from train_config import *

# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)


if __name__ == "__main__":
    # 设置需要写日志
    init_log()
    # cuda 调用
    device = get_device(sys.argv)

    param_dict = {
        'lb': lb,
        'ub': ub,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

    ### 生成训练点 ####
    # 区域内采点数
    if d==1:
        N_R = 10000
    elif d==2:
        N_R = 20000
    elif d==5:
        N_R = 50000
    else:
        # N_R = 50000
        N_R = 10000
    # lhs采样 size=[2,N_f]
    x = lb + (ub-lb)*lhs(d, N_R)


    # 打印参数
    log_str = ' d ' + str(d) + ' lambda ' + str(lambda_) + ' N_R '+str(N_R)
    log(log_str)

    # 训练参数
    train_dict = {
        'lambda_': lambda_,
        'x': x,
        'd': d,
        'N_R': N_R,
    }

    if d==1 or d==2:
        layers = [d, 20, 20, 20, 20, 1]
    elif d==5:
        layers = [d, 40, 40, 40, 40, 1]
    else:
        # layers = [d, 80, 80, 80, 80, 1]
        layers = [d, 40, 40, 40, 40, 1]
    # layers = [d, 40, 40, 40, 40, 1]
    # layers = [d, 80, 80, 80, 80, 1]
    # layers = [d, 120, 120, 120, 120, 1]
    # layers = [d, 40, 40, 40, 40, 40, 40, 1]
    # layers = [d, 80, 80, 80, 80, 80, 80, 1]
    log(layers)
    # in_num = d
    # out_num = 1
    # block_layers = [20, 20]
    # block_num = 2
    # log_str = 'in_num ' + str(in_num) + ' out_num ' + str(out_num) + ' block_layers ' + str(block_layers) + ' block_num ' + str(block_num)
    # log(log_str)

    # 训练
    # train_Adam_LBFGS(layers, device, param_dict, train_dict, Adam_steps=20000, LBFGS_steps=10000)

    # scheduler_params = {
    #     'milestones':[10000],
    #     'gamma': 0.1
    # }
    # train_Adam(layers, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-2, scheduler_name='MultiStepLR', scheduler_params=scheduler_params)
    if d==1 or d==2:
        train_Adam(layers, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3)
    elif d==5:
        train_Adam(layers, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3)
    else:
        train_Adam(layers, device, param_dict, train_dict, Adam_steps=100000, Adam_init_lr=1e-3)
    # train_Adam(layers, device, param_dict, train_dict, Adam_steps=200000, Adam_init_lr=1e-3)
    
    # train_LBFGS(layers, device, param_dict, train_dict, LBFGS_steps=20000)
    # train_Adam_ResNet(in_num, out_num, block_layers, block_num, device, param_dict, train_dict, Adam_steps=100000, Adam_init_lr=1e-3)
