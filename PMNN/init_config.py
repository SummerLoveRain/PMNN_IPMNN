import os
import os.path
import datetime
import logging
import os
import random
import time
import numpy as np
import torch
from model import MLP, ResNet
from model_config import PINNConfig

# 获取当前目录名


def getCurDirName():
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # curDirName = curDir.split(parentDir)[-1]
    curDirName = os.path.split(curDir)[-1]
    return curDirName

# 获取父目录路径


def getParentDir():
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # this will return parent directory.
    parentDir = os.path.abspath(os.path.join(curDir, os.pardir))
    # parentDirName = os.path.split(parentDir)[-1]
    return parentDir

# 固定随机种子，让每次运行结果一致


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(0)
TASK_NAME = 'task_' + getCurDirName()
now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = getParentDir() + '/data/'
path = '/' + TASK_NAME + '/' + now_str + '/'
log_path = root_path + '/' + path

# 开启日志记录功能


def init_log():
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, 'log.txt'),
                        level=logging.INFO)


# 从输入获取使用设备
def get_device(argv):
    if len(argv) > 1 and 'cuda' == argv[1] and torch.cuda.is_available(
    ):
        device = 'cuda'
    else:
        device = 'cpu'
    print('using device ' + device)
    logging.info('using device ' + device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    # device = torch.device('cpu')
    return device

# 用Adam训练


def train_Adam(layers, device, param_dict, train_dict, Adam_steps=50000,  Adam_init_lr=1e-3, scheduler_name=None, scheduler_params=None):
    # 记录时间
    start_time = time.time()
    # 默认全连接MLP
    model = MLP(layers)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用Adam训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_Adam(params=params, Adam_steps=Adam_steps, Adam_init_lr=Adam_init_lr,
                            scheduler_name=scheduler_name, scheduler_params=scheduler_params)
    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))

# 用Adam训练


def train_Adam_ResNet(in_num, out_num, block_layers, block_num, device, param_dict, train_dict, Adam_steps=50000,  Adam_init_lr=1e-3, scheduler_name=None, scheduler_params=None):
    # 记录时间
    start_time = time.time()
    # 默认全连接MLP
    model = ResNet(in_num, out_num, block_layers, block_num)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用Adam训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_Adam(params=params, Adam_steps=Adam_steps, Adam_init_lr=Adam_init_lr,
                            scheduler_name=scheduler_name, scheduler_params=scheduler_params)
    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))


# 先用Adam训练，再用LBFGS训练
def train_LBFGS(layers, device, param_dict, train_dict, LBFGS_steps=10000):
    # 记录时间
    start_time = time.time()
    # 默认全连接MLP
    model = MLP(layers)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用LBFGS训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_LBFGS(params=params, LBFGS_steps=LBFGS_steps)

    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))

# 先用Adam训练，再用LBFGS训练


def train_Adam_LBFGS(layers, device, param_dict, train_dict, Adam_steps=50000, LBFGS_steps=10000):
    # 记录时间
    start_time = time.time()
    # 默认全连接MLP
    model = MLP(layers)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用Adam训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_Adam(params=params, Adam_steps=Adam_steps)
    # 切换训练方式时，应该加载Adam最优结果
    # 加载模型
    net_path = root_path + '/' + path + '/PINN.pkl'
    model_config = PINNConfig.reload_config(net_path=net_path)
    # 用LBFGS训练
    model = model_config.model
    params = model.parameters()
    model_config.train_LBFGS(params=params, LBFGS_steps=LBFGS_steps)

    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))
