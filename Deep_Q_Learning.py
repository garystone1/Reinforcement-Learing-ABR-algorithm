import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import fixed_env as fixed_env
import load_trace as load_trace

import time as tm
import os, sys
import multiprocessing as mp

# 超参数
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 10000      # 记忆库大小

# env = gym.make('CartPole-v0')   # 立杆子游戏
# env = env.unwrapped
# N_ACTIONS = env.action_space.n  # 杆子能做的动作
# N_STATES = env.observation_space.shape[0]   # 杆子能获取的环境信息数
N_BITRATE = 4
N_REPLAY_BUFFER = 2
N_LATENCY = 10
N_ACTIONS = N_BITRATE * NREPLAY * N_LATENCY
time,time_interval, send_data_size, chunk_len,rebuf, buffer_size, play_time_len,end_delay,\
    cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
    buffer_flag, cdn_flag, skip_flag,end_of_video = 0
N_STATES = 15

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self,  time,time_interval, send_data_size, chunk_len,rebuf, buffer_size, play_time_len,end_delay,\
            cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
            buffer_flag, cdn_flag, skip_flag,end_of_video):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
      
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

    def choose_action(self, time,time_interval, send_data_size, chunk_len,rebuf, buffer_size, play_time_len,end_delay,\
        cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
        buffer_flag, cdn_flag, skip_flag,end_of_video):
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(time,time_interval, send_data_size, chunk_len,rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
                buffer_flag, cdn_flag, skip_flag,end_of_video)
            action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]     # return the argmax
        else:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def training(training_set):
    VIDEO_TRACE = testcase[0]
    NETWORK_TRACE = testcase[1]
    DEBUG = testcase[2]
    LOG_FILE_PATH = './log/'
    if not os.path.exists(LOG_FILE_PATH):
        os.makedirs(LOG_FILE_PATH)
    print(f"{VIDEO_TRACE},{NETWORK_TRACE}: Start")
    
    # -- End Configuration --
    # You shouldn't need to change the rest of the code here.
    network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
    video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
    #random_seed 
    random_seed = 2
    count = 0
    trace_count = 1
    FPS = 25
    frame_time_len = 0.04
    reward_all_sum = 0
    run_time = 0
    net_env = fixed_env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                                    random_seed=random_seed, logfile_path=LOG_FILE_PATH,
                                    VIDEO_SIZE_FILE=video_trace_prefix, Debug = DEBUG)

    BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
    TARGET_BUFFER = [0.5,1.0]   # seconds
    # ABR setting
    RESEVOIR = 0.5
    CUSHION  = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 0
    latency_limit = 4

    # QOE setting
    reward_frame = 0
    reward_all = 0
    SMOOTH_PENALTY= 0.02
    REBUF_PENALTY = 1.85
    LANTENCY_PENALTY = 0.005
    SKIP_PENALTY = 0.5
    
    while True:
        reward = 0
        # 选动作, 得到环境反馈
        next_time, next_time_interval, next_send_data_size, next_chunk_len, next_rebuf, next_buffer_size, next_play_time_len, next_end_delay,\
            next_cdn_newest_id, next_download_id, next_cdn_has_frame, next_skip_frame_time_len, next_decision_flag,\
            next_buffer_flag, next_cdn_flag, next_skip_flag, next_end_of_video = net_env.get_video_frame(bit_rate,target_buffer, latency_limit)
        # QOE setting 
        if end_delay <=1.0:
            LANTENCY_PENALTY = 0.005
        else:
            LANTENCY_PENALTY = 0.01
            
        if not cdn_flag:
            reward = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY  * end_delay - SKIP_PENALTY * skip_frame_time_len 
        else:
            reward = -(REBUF_PENALTY * rebuf)
        if decision_flag or end_of_video:
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            reward += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            last_bit_rate = bit_rate
            cnt += 1
            timestamp_start = tm.time()
            a = dqn.choose_action(time,time_interval, send_data_size, chunk_len,rebuf, buffer_size, play_time_len,end_delay,\
            cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
            buffer_flag, cdn_flag, skip_flag,end_of_video)
            bit_rate = int(a/20);
            replay_buffer = a%2
            latency_limit = int((a%20)/2)/10
            timestamp_end = tm.time()
            call_time_sum += timestamp_end - timestamp_start
        
        
        dqn.store_transition([time, time_interval, send_data_size, chunk_len,rebuf, buffer_size, play_time_len,end_delay,\
            cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
            buffer_flag, cdn_flag, skip_flag,end_of_video], a, reward, [next_time, next_time_interval, next_send_data_size, next_chunk_len, next_rebuf, next_buffer_size, next_play_time_len, next_end_delay,\
            next_cdn_newest_id, next_download_id, next_cdn_has_frame, next_skip_frame_time_len, next_decision_flag,\
            next_buffer_flag, next_cdn_flag, next_skip_flag, next_end_of_video])
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn() # 记忆库满了就进行学习
            if trace_count >= len(all_file_names):
                break

        if end_of_video:
            # print("network traceID, network_reward, avg_running_time", trace_count, reward_all, call_time_sum/cnt)
            reward_all_sum += reward_all
            run_time += call_time_sum / cnt
            
            trace_count += 1
            cnt = 0                    #algorithm processing time.
            call_time_sum = 0
            last_bit_rate = 0
            reward_all = 0
            bit_rate = 0
            target_buffer = 0
            
        time = next_time
        time_interval = next_time_interval
        send_data_size = next_send_data_size
        chunk_len = next_chunk_len
        rebuf = next_rebuf
        buffer_size = next_buffer_size
        play_time_len = next_play_time_len
        end_delay = next_end_delay
        cdn_newest_id = next_cdn_newest_id
        download_id = next_download_id
        cdn_has_frame = next_cdn_has_frame
        skip_frame_time_len = next_skip_frame_time_len
        decision_flag = next_decision_flag
        buffer_flag = next_buffer_flag
        cdn_flag = next_cdn_flag
        kip_flag = next_kip_flag
        end_of_video = next_end_of_video
        reward_all += reward
    print(f"{VIDEO_TRACE},{NETWORK_TRACE}: Done")
    return [reward_all_sum / trace_count, run_time / trace_count]

dqn = DQN() # 定义 DQN 系统
if __name__ == "__main__":
    if(sys.argv[1]=="all"):
        video_traces = [
            'AsianCup_China_Uzbekistan',
            'Fengtimo_2018_11_3', 
            'game', 
            'room', 
            'sports', 
            'YYF_2018_08_12'
        ]
        netwrok_traces = [
            'fixed',
            'low',
            'medium',
            'high'
        ]
    else:
        video_traces = [sys.argv[1]]
        netwrok_traces = [sys.argv[2]]
    debug = False
    training_set = []
    for video_trace in video_traces:
        for netwrok_trace in netwrok_traces:
            training_set.append([video_trace, netwrok_trace, debug])
    N = mp.cpu_count()
    with mp.Pool(processes=N) as p:
        results = p.map(training,training_set)
    print(results)
    print("score: ", np.mean(results ,axis = 0))
