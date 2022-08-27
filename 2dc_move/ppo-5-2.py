import os, math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#import func1
#from func1 import move

stay = 0
left = 1
up = 2
right = 3
down = 4
nrow = 12
ncol = 12
nstate = nrow * ncol
lr = 0.001
node_loc = []
loc_num = [416, 1250, 2083, 2916, 3750, 4583, 5416, 6250, 7083, 7916, 8750, 9583]
action_space = []
#memorysize = 600
#itertimes = 20
epsilon = 1
gamma = 0.9

#train
memorysize = 5000
itertimes = 20
batch_size = 32

#debug
#memorysize = 50
#itertimes = 5
#batch_size = 8

loss_val = []

for i in range(1,6):
    for j in range(5):
        for k in range(5):
            action_space.append([i,j,k])


naction = len(action_space)



for i in range(nrow):
    for j in range(ncol):
        node_loc.append([loc_num[i],loc_num[j]])

def readresult(filename):
    a1 = np.genfromtxt(filename)
    return a1


def readvdi(file):  # 读取csv中的vdi的数据，得到的是array数据，这里作为input
    zvdi = readresult(file)
    z = zvdi[:, 2]
    return z

def move(action, cap_idx):
    while 1:
        if action == 0:
            break
        if action == 1:
            if cap_idx % 12 == 0:
                if cap_idx == 0:
                    action = random.choice([3, 2, 0])
                    continue
                if cap_idx == 132:
                    action = random.choice([3, 4, 0])
                    continue
                action = random.choice([3, 2, 0, 4])
                continue
            cap_idx -= 1
            break
        if action == 3:
            if cap_idx % 12 == 11:
                if cap_idx == 11:
                    action = random.choice([1, 2, 0])
                    continue
                if cap_idx == 143:
                    action = random.choice([1, 4, 0])
                    continue
                action = random.choice([1, 2, 0, 4])
                continue
            cap_idx += 1
            break

        if action == 2:
            if 132 < cap_idx < 143:
                action = random.choice([3, 4, 1, 0])
                continue
            if cap_idx == 143:
                action = random.choice([1, 4, 0])
                continue
            if cap_idx == 132:
                action = random.choice([3, 4, 0])
                continue
            cap_idx += 12
            break

        if action == 4:
            if 0 < cap_idx < 11:
                action = random.choice([3, 2, 1, 0])
                continue
            if cap_idx == 0:
                action = random.choice([1, 2, 0])
                continue
            if cap_idx == 11:
                action = random.choice([3, 2, 0])
                continue
            cap_idx -= 12
            break

    return action, cap_idx

def location(file):  # 节点位置
    a = readresult(file)
    x = a[:, 0]
    y = a[:, 1]
    loc = []
    for r in range(len(a)):
        loc.append([x[r], y[r]])
    return loc


# 运行程序获得VDI分布图,即得到csv数据
def run_os():
    os.system('ngspice -b interposer1_tr.sp -r interposer1_tr.raw')
    os.system('bin/inttrvmap int1.conf interposer1_tr.raw 1.0 0.05')


def target_vdi():
    os.system('bin/diedcapgen 10 1e-9 chiplet1_vdd.decap vdd_decap.1')
    run_os()
    a = readvdi('chiplet1_vdd_1_vdi.csv')
    sum_a = np.sum(a)
    return sum_a


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(nstate, 500)
        self.fc2 = nn.Linear(500, 500)
        self.out = nn.Linear(500, len(action_space))

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x2 = self.fc2(x1)
        x2 = F.relu(x2)
        out = self.out(x2)
        return out


class PPO:
    def __init__(self, ):
        self.eval_net, self.target_net = Net(), Net()
        self.memorysize = memorysize
        self.memory = np.zeros([memorysize, nstate * 2 + 2])
        self.mem_cnt = 0
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.learn_step_counter = 0

    def csact(self, x):  # 选择动作函数
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if random.random() < epsilon:
            action = random.randint(0, naction - 1)
        else:
            action_val = self.eval_net.forward(x)
            action = torch.max(action_val, 1)[1].data.numpy()
            action = action[0]
        return action

    def step(self, cap_idx, action_idx,decap_val,):  # 这里action应该是0-4，代表五个方向,decap_val是可选电容值
        str1 = ''
        for i in cap_idx:
            str1 += 'c_decap_%d_%d nd_1_0_%d_%d 0 %e\n' % (cap_idx.index(i),i,node_loc[i][0],node_loc[i][1],decap_val)

        f = open('vdd_decap.1','w')
        f.write(str1)
        f.close()
        run_os()
        state_ = readvdi('chiplet1_vdd_1_vdi.csv')
        return state_, action_idx, cap_idx

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # replace the old memory with new memory
        index = self.mem_cnt % memorysize
        self.memory[index, :] = transition
        self.mem_cnt += 1

    def reset(self, decap_val):  # cap是随机增加的cap数目，val是每个cap的容值
        cap_idx = random.sample([a for a in range(nstate)],10)
        str1 = ''
        for i in cap_idx:
            str1 += 'c_decap_%d_%d nd_1_0_%d_%d 0 %e\n' % (cap_idx.index(i),i,node_loc[i][0], node_loc[i][1], decap_val)
        with open('vdd_decap.1', 'w') as f1:
            f1.write(str1)
        f1.close()
        run_os()
        state = readvdi('chiplet1_vdd_1_vdi.csv')

        return state,cap_idx

    def learn(self):
        # target parameter update
        if self.learn_step_counter % itertimes == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample batch transitions
       # sample_index = np.random.choice(memorysize, batch_size)
        sample_index = np.random.choice(min(memorysize,self.mem_cnt),batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :nstate])
        b_a = torch.LongTensor(b_memory[:, nstate:nstate + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, nstate + 1:nstate + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -nstate:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        loss_val.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#train
episodes = 8000

#debug
#episodes = 1000




chip = PPO()
reward = []
act = []
total_vdis = []
cap_idx_list = []
epsilons = []
for eps in range(episodes):
    if eps == 0 :
        s,cap_idx = chip.reset(1e-9)
        ccc = cap_idx
        f = open('vdd_decap.1','r')
        st = f.readlines()
        f.close()
        target = 1.5e-10
    if eps % 800 == 0:
        f1 = open('vdd_decap.1','w')
        for i in range(10):
            f1.write(st[i])
        f1.close()
        run_os()
        s = readvdi('chiplet1_vdd_1_vdi.csv')
        cap_idx = ccc

    a = chip.csact(s)
    s_,a,cap_idx1 = chip.step(cap_idx,a,1e-9)
    r = (1.e-10 - np.sum(s_)) / 1.e-10 
    chip.store_transition(s,a,r,s_)
    if chip.mem_cnt >0.6* memorysize:
        chip.learn()
    reward.append(r)
    total_vdis.append(np.sum(s_))
    s = s_
    cap_idx_list.append(cap_idx1)
    act.append(a)
    if eps < 0.9 *  memorysize:
        epsilon = 0.9
    else:
        epsilon = 0.9* np.exp( -0.002 * (eps -memorysize) )
    cap_idx = cap_idx1
    epsilons.append(epsilon)
    #epsilon = np.exp(-3 * eps / episodes)

loss_list = torch.tensor(loss_val)
cap_array = np.array(cap_idx_list)
cap_move = cap_array[:100,:]
#print(cap_move)
#o = input('cap change')
plt.plot([p for p in range(len(cap_idx_list))],cap_idx_list)
plt.title('cap list')
plt.figure()
plt.plot([a for a in range(len(loss_list))],loss_list)
plt.title('loss funtion')
plt.xlabel('episodes')
plt.ylabel('value')
plt.figure()
plt.plot([b for b in range(len(reward))],reward)
plt.title('reward')
plt.figure()
plt.plot([c for c in range(len(total_vdis))],total_vdis)
plt.plot([0,episodes - 1],[1.5e-10,1.5e-10],label='target VDI',linestyle='--')
plt.title('VDI plot')
plt.legend()
plt.figure()
plt.subplot(1,2,1)
plt.plot([m1 for m1 in range(len(cap_move))],cap_move[:,0],label='no.0')
plt.plot([m2 for m2 in range(len(cap_move))],cap_move[:,1],label='no.1')
plt.plot([m3 for m3 in range(len(cap_move))],cap_move[:,2],label='no.2')
plt.plot([m4 for m4 in range(len(cap_move))],cap_move[:,3],label='no.3')
plt.plot([m5 for m5 in range(len(cap_move))],cap_move[:,4],label='no.4')
plt.subplot(1,2,2)
plt.plot([m6 for m6 in range(len(cap_move))],cap_move[:,5],label='no.5')
plt.plot([m7 for m7 in range(len(cap_move))],cap_move[:,6],label='no.6')
plt.plot([m8 for m8 in range(len(cap_move))],cap_move[:,7],label='no.7')
plt.plot([m9 for m9 in range(len(cap_move))],cap_move[:,8],label='no.8')
plt.plot([m10 for m10 in range(len(cap_move))],cap_move[:,9],label='no.9')

plt.legend()
plt.show()


torch.save(target_net.state_dict(),'target_net_model.pkl')      # save agent parameters
# torch.save(target_net,'target_net.pkl')                           whole agent net
# net1 = torch.load('target_net.pkl')                               load net







#w = input('test:')
f1 = open('vdd_decap.1','w')
for i in range(10):
    f1.write(st[i])
f1.close()
run_os()
s = readvdi('chiplet1_vdd_1_vdi.csv')
cap_idx = ccc
epsilon = 0
print(cap_idx)
x = input('cap_idx')
test_reward = []
test_cap_idx = []
action = []
for u in range(50):
    a = chip.csact(s)
    s_,a,cap_idx1 = chip.step(cap_idx,a,1e-9)
    r = (1.5e-10 - np.sum(s_)) / (1.5e-10 + np.sum(s_))
    test_reward.append(r)
    test_cap_idx.append(cap_idx)
    s = s_
    action.append(a)
    cap_idx = cap_idx1

cap_array1 = np.array(test_cap_idx)
cap_move1 = cap_array1[:30,:]



plt.plot([a for a in range(len(test_reward))],test_reward)
plt.title('reward')
plt.figure()

plt.plot([b for b in range(50)],action)
plt.title('decap action')

plt.figure()
plt.plot([b for b in range(len(test_cap_idx))],test_cap_idx)
plt.title('decap location')
plt.figure()
plt.subplot(1,1,1)
plt.plot([m1 for m1 in range(len(cap_move1))],cap_move1,label='no.1')
#plt.plot([m2 for m2 in range(len(cap_move1))],cap_move1[:,1],label='no.2')
plt.legend()
plt.show()
print(cap_idx)
