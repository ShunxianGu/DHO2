import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os

from dataloader import cifar10_dataset, cifar100_dataset
import torchvision
from typing import Optional

from torch._functorch.make_functional import make_functional_with_buffers

import torch.nn as nn
import torchopt

import torch.distributed as dist

from time import time

from copy import deepcopy
from fosi.torch_optim.extreme_spectrum_estimation import get_ese_fn
# from utils import sparse_top_k, sparse_randomized
import cifar_resnet as models

# from DenseNet import DenseNet201
from VGG import VGG16
# from VGGG import VGG13
# from ResNet import resnet152

CPU_ITER = 1
# num_cpus = torch.get_num_threads()
# world_size = torch.cuda.device_count() # gpu number + cpu
k_num = 16 #resnet 152
decay = 0.9
# RATIO = 0.5
# num_cpus = torch.get_num_threads()

from imagenet import train_dataset, val_dataset , test_dataset



def ddp_setup():
    """
    setup the global group
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl")

    # all device should use cuda
    # torch.cuda.set_device(rank)

class Worker:
    def __init__(
        self,
        rank: int,
        model: torch.nn.Module,
        data: tuple,
        gpu_id: int,
        optimizer: torchopt.base.GradientTransformation,
        alpha: float = 0.01,
        approx_k: int = 8,
        approx_l: int = 0,
        learning_rate_clip: Optional[float] = 1.0
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.model, self.params, self.buffers = make_functional_with_buffers(self.model)

        broadcast_params = torch.nn.utils.parameters_to_vector(self.params)
        # broadcast_params = torch.zeros_like(torch.nn.utils.parameters_to_vector(self.params))
        # print(broadcast_params.shape)
        # assert 0
        dist.broadcast(broadcast_params, 0)
        torch.nn.utils.vector_to_parameters(broadcast_params, self.params)
        self.pi = torch.zeros_like(torch.nn.utils.parameters_to_vector(self.params))
        
        self.train_data = data[0]
        self.test_data = data[1]
        self.rank = rank

        # self.base_optimizer = optimizer
        self.optimizer = optimizer
        self.alpha = alpha
        self.init_fn()

        if self.rank == 0:
            self.acc_list = []
            self.train_iter_time = []

###################################################
        # self.test_data = data[1] # into vram

        self.ese_fn = get_ese_fn(self.loss_fn_batch, approx_k, next(iter(self.train_data)), approx_l, precondition_world_size = world_size, worker_world_size = 0, group = None, rank = self.rank)

        self.approx_l = approx_l
        self.learning_rate_clip = learning_rate_clip
        self.device = rank

        self.sigma = 5e-9 #5e-5:82 5e-6 


    def init_fn(self):
        flatten_params = torch.nn.utils.parameters_to_vector(self.params)
        num_params = flatten_params.shape[0]
        
        self.opt_state = self.optimizer.init(flatten_params)
        self.velocity = torch.zeros((num_params,), dtype=torch.int32).to(self.gpu_id)

        self.k_learning_rates = torch.zeros((k_num,), dtype=torch.float32).to(self.gpu_id)
        self.k_eigenvecs = torch.zeros((k_num, num_params), dtype=torch.float32).to(self.gpu_id)
        
        self.scaling_factor = torch.ones((1,), dtype=torch.float32).to(self.gpu_id)
        
        self.momentum_func = lambda g, t: (1 - decay) * g + decay * t


    def loss_fn_batch(self,params, batch):
        preds = self.model(params, self.buffers, batch[0].cuda())
        loss = nn.CrossEntropyLoss()(preds, batch[1].cuda())
        return loss

    def dist_lanczos(self):
        spare_num = world_size - (self.k_eigenvecs.shape[1] % world_size)
        # print(spare_num)
        num_params = (self.k_eigenvecs.shape[1]+spare_num) // world_size
        # num_items = self.k_learning_rates.shape[0]
        
        k_eigenvals, k_eigenvecs = self.ese_fn(self.params)

        # if self.rank == 15:
        #     print(k_eigenvals)

        # print(k_eigenvecs.shape)
        k_learning_rates = torch.abs(1.0 / (k_eigenvals + self.sigma))
        self.scaling_factor = torch.clip(k_learning_rates[self.approx_l] / k_learning_rates[-1], 1.0, self.learning_rate_clip)
        # var_list = list(torch.split(k_eigenvecs, k_num // world_size, dim=0))
        # print(k_eigenvecs)
        # print(var_list[0])
        # if self.rank == 15:
        #     print(k_eigenvecs[:,:-spare_num])
        
        # assert 0

        contigious_cache = self.k_eigenvecs[:,0*num_params:(0+1)*num_params].contiguous()

        for i in range(world_size):
            contigious_cache.zero_()
            contigious_cache.copy_(k_eigenvecs)
            if i == self.device:
                dist.broadcast(contigious_cache,  src=self.device, async_op=False)
            else:
                dist.broadcast(contigious_cache, src=i, async_op=False)
                # if i == 7 and self.device==0:
                #     print(contigious_cache)
                #     assert 0


            if i == (world_size-1) and spare_num != 0:
                
                # print(contigious_cache[:,:-spare_num])
                self.k_eigenvecs[:,i*num_params:(i+1)*num_params-spare_num].copy_(contigious_cache[:,:-spare_num])
            else:
                self.k_eigenvecs[:,i*num_params:(i+1)*num_params].copy_(contigious_cache)

        self.k_learning_rates.copy_(k_learning_rates)

        # print()

        # if self.rank == 15 or self.rank == 0:
        #     print(self.k_eigenvecs)
        #     print(spare_num)
        #     assert 0
        # assert 0



    def loss_fn(self, source, targets):
        preds = self.model(self.params, self.buffers, source)
        loss = nn.CrossEntropyLoss()(preds, targets)
        return loss

    def accuracy(self, source, targets):
        preds = self.model(self.params, self.buffers, source)
        predicted_class = torch.argmax(preds, dim=1)
        return torch.sum(predicted_class == targets)


    def get_g(self, updates):
        g = updates
        g1 = torch.matmul(torch.matmul(self.k_eigenvecs, g), self.k_eigenvecs)
        return g1

    def _appprox_newton_direction(self, g1):
        newton_direction = torch.matmul(self.k_learning_rates * torch.matmul(self.k_eigenvecs, g1), self.k_eigenvecs)
        # dist.all_reduce(newton_direction, op=dist.reduce_op.SUM, async_op=False)
        return newton_direction

    def _orthogonalize_vector_wrt_eigenvectors(self,v):
        v_ = torch.matmul(torch.matmul(self.k_eigenvecs, v), self.k_eigenvecs)
        # dist.all_reduce(v_, op=dist.reduce_op.SUM, async_op=False)
        v = v - v_
        return v


    def newton(self,updates, g1):
        
        g = updates
        g2 = g - g1
        
        flatten_params = torch.nn.utils.parameters_to_vector(self.params)

        self.velocity = self.momentum_func(g1, self.velocity)

        newton_direction = self._appprox_newton_direction(self.velocity)

        base_opt_deltas, self.opt_state = self.optimizer.update(g2, self.opt_state, params=flatten_params)
        base_opt_deltas = self._orthogonalize_vector_wrt_eigenvectors(base_opt_deltas)
        # torch.nn.utils.vector_to_parameters(self.scaling_factor * base_opt_deltas - self.alpha * newton_direction, updates)

        # return updates
        return self.scaling_factor * base_opt_deltas - self.alpha * newton_direction


    def update_step(self, grad):

        Grad_temp = torch.nn.utils.parameters_to_vector(grad) + self.pi
        
        g1 = self.get_g(Grad_temp)
        # dist.all_reduce(g1, op=dist.reduce_op.SUM, async_op=False)
        updates = self.newton(Grad_temp,g1)

        # self.pi += (self.sigma * updates)

        # updates += (self.pi/self.sigma)

        torch.nn.utils.vector_to_parameters(updates, grad)

        self.params = torchopt.apply_updates(self.params, grad, inplace=True)

    def synchronize_params(self):
        delta = torch.nn.utils.parameters_to_vector(self.params).clone().detach() - self.cache_param
        self.pi += (self.sigma * delta)
        zelta = torch.nn.utils.parameters_to_vector(self.params).clone().detach() + self.pi / self.sigma


        ################################################################
        # dist.all_reduce(zelta, op=dist.reduce_op.SUM, async_op=False)
        # zelta /= world_size
        ################################################################

        # dist.all_reduce(self.accum_params, op=dist.reduce_op.SUM, async_op=False)
        # self.checkpoints.copy_(self.accum_params)
        torch.nn.utils.vector_to_parameters(zelta.clone().detach(), self.params)

        
    def _run_batch(self, source, targets, epoch):


        loss = self.loss_fn(source,targets)
        grads = torch.autograd.grad(loss, self.params)
        flatten_grads = torch.nn.utils.parameters_to_vector(grads)

        # flatten_grads = sparse_randomized(flatten_grads)

        dist.all_reduce(flatten_grads, op=dist.ReduceOp.SUM, async_op=False)
        flatten_grads = flatten_grads / world_size

        # if epoch == 0:
        #     base_opt_deltas, self.opt_state = self.optimizer.update(flatten_grads, self.opt_state)
        #     torch.nn.utils.vector_to_parameters(base_opt_deltas, grads)
        #     self.params = torchopt.apply_updates(self.params, grads, inplace=True)

        # else:
        torch.nn.utils.vector_to_parameters(flatten_grads, grads)
        self.update_step(grads)


    def compute_accuracy(self):

        acc = 0.0
        num_samples = 0
        for source, targets in self.test_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            acc += self.accuracy(source, targets)
            num_samples += source.shape[0]


        acc /= num_samples
        dist.all_reduce(acc, op=dist.ReduceOp.SUM, async_op=False)
        acc /= world_size
        print(f'Test accuracy: {acc}')

        if self.rank == 0:
            self.acc_list.append(acc.cpu())

        # if self.rank == 0 and acc >= 0.917: #resnet baseline:
        #     import numpy as np
        #     np.save("resnet110_DHO2_917.npy", np.array(self.acc_list))
        #     per_iter_time = sum(self.train_iter_time) / len(self.train_iter_time)
        #     print(f'Training iteration time: {per_iter_time}')
        #     # assert 0

        # if self.rank == 0 and acc >= 0.919: #resnet baseline:
        #     import numpy as np
        #     np.save("resnet110_DHO2_919.npy", np.array(self.acc_list))
        #     per_iter_time = sum(self.train_iter_time) / len(self.train_iter_time)
        #     print(f'Training iteration time: {per_iter_time}')

        # if self.rank == 0 and acc >= 0.85: #resnet baseline:
        #     import numpy as np
        #     np.save("densenet_G32_DHO2_92.npy", np.array(self.acc_list))
        #     per_iter_time = sum(self.train_iter_time) / len(self.train_iter_time)
        #     print(f'Training iteration time: {per_iter_time}')
        #     assert 0
        

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)

            # if self.rank == 0:
            #     start_iter_time = time()

            self._run_batch(source, targets,epoch)

        

    def train(self, max_epochs: int):

        for epoch in range(max_epochs):
                
            a = time()
            if ((epoch) % 8 == 0):
                self.dist_lanczos()

            if ((epoch) % 8 == 0):
                self.cache_param = torch.nn.utils.parameters_to_vector(self.params).clone().detach()
                self.pi_change = 0
            self._run_epoch(epoch)
            self.pi_change +=1
            # if ((epoch) % 4 == 0):
            # if epoch == 0:

            #     self.pi = torch.zeros_like(torch.nn.utils.parameters_to_vector(self.params))
                # self.pi = torch.nn.utils.parameters_to_vector(self.params).clone().detach()


            b = time()
            print("one epoch time is:"+str(b-a))   
            self.compute_accuracy()
            if self.rank == 0:
                self.train_iter_time.append((b-a)/len(self.train_data))
                
            if self.pi_change == 8:
                # assert 0
                self.synchronize_params()
                # self.compute_accuracy()

        if self.rank == 0:
            import numpy as np
            
            np.save("resnet_152_ADMM_original_64.npy", np.array(self.acc_list))
            per_iter_time = sum(self.train_iter_time) / len(self.train_iter_time)
            print(f'Training iteration time: {per_iter_time}')



def load_train_objs():
    # dataset_base_path = '/dev/shm/cifar'
    dataset_base_path = '/data/public/cifar'
    # dataset_base_path = os.path.join("datasets", "cifar10")
    # train_set = cifar100_dataset(dataset_base_path, train_flag=True)
    # test_set = cifar100_dataset(dataset_base_path, train_flag=False)

    train_set = train_dataset

    test_set = val_dataset

    # model = torchvision.models.resnet50(num_classes = 10)  # load your model here
    # model = models.get_model("resnet110")
    # model = torchvision.models.vit_b_16()
    # model = torchvision.models.swin_s()  # load your model here

    # model.head = nn.Linear(768, 10)
    # model = DenseNet201()

    # model = VGG16()
    # pretrain_model = torchvision.models.swin_b(weights = "Swin_B_Weights.IMAGENET1K_V1")  # load your model here
    # class linear_forward(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #     def forward(self, x):
    #         return x
    # pretrain_model.head = linear_forward()

    # class Classifier(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.linear_1 = nn.Linear(1024, 100, bias=True)
    #         # self.activation_1 = nn.ReLU()
    #         # self.linear_2 = nn.Linear(120, num_ftrs, bias=False)


    #     def forward(self, x):

    #         # self.temp_1 = self.activation_1(self.linear_1(x))
    #         output = self.linear_1(x)

    #         return output #self.temp_1

    model = torchvision.models.resnet152(num_classes = 200)

    base_optimizer = torchopt.adam(lr=28e-4)
    
    return (train_set,test_set), model, base_optimizer


def prepare_dataloader(dataset: tuple, batch_size: int):

    return (DataLoader(
        dataset[0],
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset[0])
    ),DataLoader(
        dataset[1],
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset[1]),
        drop_last = True
    ))

def main(rank: int, local_rank: int,  world_size: int, save_every: int, total_epochs: int, batch_size: int):
    # ddp_setup(rank, world_size)
    
    dataset, model, optimizer = load_train_objs()
    # print("XXXXXXXXXXXXXXXXXXXXXXXX")
    data = prepare_dataloader(dataset, batch_size)

    torch.cuda.set_device(local_rank)

    trainer = Worker(rank,model, data, local_rank, optimizer,0.028, k_num) 
    trainer.train(total_epochs)

    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs',default=100, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--local-rank', type=int)
    ddp_setup()
    args = parser.parse_args()
    global world_size
    print("preconditioner num is:"+str(k_num))

    world_size = dist.get_world_size()

    print("world is:"+str(world_size))
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    print(local_rank)
    main(rank, local_rank, world_size, 0, args.total_epochs, args.batch_size)
