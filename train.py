import tempfile
import time
import math
from torch import nn

from sklearn import metrics
import argparse
from models import LSTM,ASCNN,ASGCN ,ASBIGCN
from data__process import *
import torch
import random
import numpy as np
import pickle
from diy_iterator import Bucket_Iterator
from optimization import BertAdam
from torch.utils.data import DataLoader
import torch.distributed as dist

torch.set_printoptions(profile="full")
#bert_model= BertModel.from_pretrained('bert-base-uncased')

class Instructor(object):
    def __init__(self,jiamao,args):
        self.jiamao=jiamao
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.args=args
        init_distributed_mode(args=self.args)
        self.rank = self.args.rank
        self.jiamao.device = torch.device(self.args.device)
        self.jiamao.learning_rate *= self.args.world_size  # 学习率要根据并行GPU的数量进行倍增
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        self.tag=0
        data_set=pickle.load(open(self.jiamao.file,'rb'))
        self.train_data_set = Bucket_Iterator(data=data_set.train_data)
        self.test_data_set = Bucket_Iterator(data=data_set.test_data)

#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data_set)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_data_set)
        self.train_batch_sampler = torch.utils.data.BatchSampler(self.train_sampler, jiamao.batch_size,drop_last=True)
        nw = min([os.cpu_count(), jiamao.batch_size if jiamao.batch_size > 1 else 0, 8])  # number of workers
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        self.train_data = DataLoader(dataset=self.train_data_set,
                                     batch_sampler=self.train_batch_sampler,
                                     num_workers=nw,
                                     pin_memory=True)

        self.test_data = DataLoader(dataset=self.test_data_set,
                                    batch_size=jiamao.batch_size,
                                    sampler=self.test_sampler,
                                    pin_memory=True,
                                    num_workers=nw,
                                    drop_last=True)

        self.model = jiamao.model_class(data_set.embedding_matrix,self.jiamao)
        self.model=self.model.to(self.jiamao.device)
        print(self.jiamao.device)

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if self.rank == 0:
            torch.save(self.model.state_dict(), checkpoint_path)
        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.jiamao.device))
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu],find_unused_parameters=True)
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        self.print_args()
        self.global_f1 = 0.

    def run(self,repeats=1):
        loss=nn.CrossEntropyLoss()
        loss=loss.to(self.jiamao.device)
        opti=self.jiamao.optimizer(self.model.parameters(),lr=self.jiamao.learning_rate,weight_decay=self.jiamao.weight_decay)
        if not os.path.exists('log/'):
            os.mkdir('log/')
        log_file = open('log/'+self.jiamao.project_name+'.txt', 'a', encoding='utf-8')
        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats):
            print('repeat: ', (i+1))
            log_file.write("repeat:"+str(i+1))
            self.reset_params()
            #self.model.bert=bert_model
            if self.jiamao.mode=="train":
                parameters = [para for  para in self.model.parameters()]
                named_params = [(name, p) for name, p in self.model.named_parameters()]
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                opti_bert = BertAdam(optimizer_grouped_parameters, lr=0.00005, warmup=0.1)
                max_test_acc, max_test_f1 = self.jiamao_train(loss,opti_bert)
                print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
                log_file.write('max_test_acc: {0}, max_test_f1: {1}'.format(max_test_acc, max_test_f1))
                max_test_acc_avg += max_test_acc
                max_test_f1_avg += max_test_f1
                print('#' * 100)
                print("max_test_acc_avg:", max_test_acc_avg / repeats)
                print("max_test_f1_avg:", max_test_f1_avg / repeats)
            else:
                self.model.load_state_dict(torch.load('state_dict' + self.jiamao.project_name + "_" + ".pkl"))
                test_acc, test_f1 = self.get_acc_and_f1_plus()
                print("max_test_acc_avg:", test_acc / repeats)
                print("max_test_f1_avg:", test_f1 / repeats)
            log_file.write("\n")
            log_file.write("==========================================================="+"\n")
        log_file.close()

    def jiamao_train(self,loss,opti_bert):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0  #### yong yu  tiqian jieshu
        for epoch in range(self.jiamao.epoch_number):

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            self.train_sampler.set_epoch(epoch=epoch)
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            print('>' * 100)
            print('epoch: ', epoch)
            num_correct, num_total = 0, 0
            increase_flag = False
            for index,sample in enumerate(self.train_data):
                self.model.train()
                global_step += 1
                opti_bert.zero_grad()
                inputs = [data_class.to(self.jiamao.device)  for data_class in sample]
                targets = inputs[2].to(self.jiamao.device)
                outputs = self.model(inputs)
                loss_res=loss(outputs,targets)
                loss_res.backward()

                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                loss=reduce_value(loss,average=True)
                # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

                opti_bert.step()
                #if global_step % self.jiamao.step==0:
                if self.jiamao.epoch_number-1==epoch and index==len(self.train_data)-1:
                    num_correct+=(torch.argmax(outputs,-1)==targets).sum().item()
                    num_total+=len(outputs)
                    train_acc=num_correct/num_total
                    test_acc,test_f1=self.get_acc_and_f1()
                    if test_acc>max_test_acc:
                        max_test_acc=test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        if self.jiamao.save and test_f1>self.global_f1:
                            self.global_f1=test_f1
                            torch.save(self.model.state_dict(),'state_dict'+self.jiamao.project_name+"_"+".pkl")
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss_res.item(), train_acc, test_acc, test_f1))
            # 等待所有进程计算完毕

            if jiamao.device != torch.device("cpu"):
                torch.cuda.synchronize(jiamao.device)
            #end train |||  in epoch
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 4:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1

    def get_acc_and_f1(self):
        self.model.eval()
        num_test_correct, num_test_total = 0, 0
        test_targets_all, test_outputs_all = None, None
        with torch.no_grad():
            for test_batch, test_sample_batched in enumerate(self.test_data):
                test_inputs = [data_class.to(self.jiamao.device)  for data_class in test_sample_batched]
                test_targets = test_inputs[2]
                test_outputs = self.model(test_inputs)
                num_test_correct += (torch.argmax(test_outputs, -1) == test_targets).sum().item()
                num_test_total += len(test_outputs)
                if test_targets_all is None:
                    test_targets_all=test_targets
                    test_outputs_all=test_outputs
                else:
                    test_targets_all=torch.cat((test_targets_all,test_targets),dim=0)
                    test_outputs_all=torch.cat((test_outputs_all,test_outputs),dim=0)
            num_test_correct = reduce_value(num_test_correct, average=False)
        test_acc = num_test_correct / num_test_total
        f1=metrics.f1_score(test_targets_all.cpu(),torch.argmax(test_outputs_all,-1).cpu(),labels=[0, 1, 2], average='macro')

        return test_acc,f1

    def get_acc_and_f1_plus(self):
        self.model.eval()
        if not os.path.exists('results/'+self.jiamao.project_name):
            os.makedirs('results/'+self.jiamao.project_name)
            os.makedirs('results/'+self.jiamao.project_name+'/right')
            os.makedirs('results/'+self.jiamao.project_name+'/false')
        num_test_correct, num_test_total = 0, 0
        test_targets_all, test_outputs_all = None, None
        all_results=[]
        with torch.no_grad():
            for test_batch, test_sample_batched in enumerate(self.test_data):
                test_inputs = [data_class.to(self.jiamao.device) for data_class in test_sample_batched]
                test_targets = test_inputs[2]
                test_outputs = self.model(test_inputs)
                num_test_correct += (torch.argmax(test_outputs, -1) == test_targets).sum().item()
                num_test_total += len(test_outputs)
                if test_targets_all is None:
                    test_targets_all=test_targets
                    test_outputs_all=test_outputs
                else:
                    test_targets_all=torch.cat((test_targets_all,test_targets),dim=0)
                    test_outputs_all=torch.cat((test_outputs_all,test_outputs),dim=0)
                for i in range(test_sample_batched['polarity'].size(0)):
                    tmpdict = {}            ###################yihou you jihui  hui lai zai kan
                    tmpdict['truelabel'] = test_targets[i].cpu().data.tolist()
                    tmpdict['predictlabel'] = torch.argmax(test_outputs, -1)[i].cpu().data.tolist()
                    tmpdict['isright'] = tmpdict['truelabel'] == tmpdict['predictlabel']
                    tmpdict['text'] = test_sample_batched['text'][i]
                    tmpdict['aspect'] = test_sample_batched['aspect'][i]
                    tmpdict['span'] = test_sample_batched['span_indices'][i]
                    tmplen = len(test_sample_batched['text'][i])
                    tmpdict['att'] = self.model.attss[1].cpu().data.numpy()[i][:tmplen, :tmplen]
                    tmpdict['att1'] = self.model.attss[2].cpu().data.numpy()[i][:tmplen, :tmplen]
                    adjs = self.model.attss[0]
                    tmpdict['adj'] = adjs[i].cpu().data.numpy()[:tmplen, :tmplen]
                    all_results.append(tmpdict)
        pickle.dump(all_results, open('results/'+self.jiamao.project_name+'/hhd.result', 'wb'))

        num_test_correct = reduce_value(num_test_correct, average=False)
        test_acc = num_test_correct / num_test_total
        f1 = metrics.f1_score(test_targets_all.cpu(), torch.argmax(test_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.jiamao):
            print('>>> {0}: {1}'.format(arg, getattr(self.jiamao, arg)))

    def reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.jiamao.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'gloo'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    print(args.dist_backend)
    print(args.dist_url)
    print(args.world_size)
    print(args.rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


if __name__=="__main__":
    start_time=time.time()
    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
        'asbi': ASBIGCN,
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    input_mode = {
        'lstm': ['text_indices'],
        'ascnn': ['cattext_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['cattext_indices', 'aspect_indices', 'left_indices', 'undir_dependency_graph'],
        'asbi': ['cattext_indices','aspect_indices','left_indices','span_out','undir_dependency_graph']#,'dependency_graph1','dependency_graph2','dependency_graph3'],#'tran_indices'
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    file={
        "first":".\\newDGEDT\\data\\project.pkl"
    }
    mode="asbi"
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world_size', default=4, type=int,help='number of distributed processes')
    parser.add_argument('--syncBN', type=bool, default=True) ##可以不使用，这个对训练速度有影响
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()
    class JiaMao():
        def __init__(self,model_classes,initializers,optimizers,file):
            self.model_class=model_classes[mode]
            self.input_mode=input_mode[mode]
            self.optimizer=optimizers
            self.initializer=initializers
            self.file=file
            self.input_dim=300
            self.hidden_dim=100
            self.polarities_dim=3
            self.learning_rate=0.001
            self.device=torch.device("cuda")
            self.seed=776
            self.drop_out=0.3
            self.batch_size=128
            self.weight_decay=0.00001
            self.project_name= "jiamao"
            self.mode="train"
            self.epoch_number=2
            self.step= 5
            self.save=True
            self.repeats=1
    jiamao=JiaMao(model_classes,initializers["xavier_uniform_"],optimizers["adam"],file["first"])

    if jiamao.seed is not None:
        seed=jiamao.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins=Instructor(jiamao,opt)
    ins.run(jiamao.repeats)
    end_time=time.time()
    print(end_time-start_time)
