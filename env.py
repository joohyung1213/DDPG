import numpy as np
import random
from argparse import ArgumentParser
import math
import tensorflow as tf
from scipy.spatial import distance
"""
Hyperparameters of environment
Contains seed, dimensions of matrices, power of transmitter and so forth.
"""
parser = ArgumentParser()
parser.add_argument("--SEED_VALUE", default = 7777777,
                    help="The value of seed value for random computation")
# number of base stations and VUEs
parser.add_argument("--NUM_MABS", default = 1,
                    help="The number of Macro cell base station (MaBS)")
parser.add_argument("--NUM_MIBS", default = 10,
                    help="The number of Micro cell base station (MiBS)")
parser.add_argument("--NUM_PBS", default = 50,
                    help="The number of Pico cell base station (PBS, mmWave based)")
parser.add_argument("--NUM_VUE", default=100,
                    help="The number of VUEs")
# coverage region of each base station
parser.add_argument("--CELL_MABS_COVERAGE", default=3000,
                    help="The cell coverage of macro cell base station (in meter)")
parser.add_argument("--CELL_MIBS_COVERAGE", default=500,
                    help="The cell coverage of micro cell base station (in meter)")
parser.add_argument("--CELL_PBS_COVERAGE", default=100,
                    help="The cell coverage of pico cell base station (in meter)")
# power of each base station
parser.add_argument("--POWER_MABS", default=40,
                    help="The TX power of MaBS (in dBm)")
parser.add_argument("--POWER_MIBS", default=35,
                    help="The TX power of MiBS (in dBm)")
parser.add_argument("--POWER_PBS", default=20,
                    help="The TX power of PBS (in dBm)")
# number of channels of each base station
parser.add_argument("--NUM_CHANNELS_MABS", default=30,
                    help="The number of channels in MaBS")
parser.add_argument("--NUM_CHANNELS_MIBS", default=30,
                    help="The number of channels in MiBS")
parser.add_argument("--NUM_CHANNELS_PBS", default=5,
                    help="The number of channels in PBS")
# bandwidth of each base station
parser.add_argument("--BANDWIDTH_MABS", default=0.18,
                    help="The bandwidth of MaBS (in Mega Hertz)")
parser.add_argument("--BANDWIDTH_MIBS", default=0.18,
                    help="The bandwidth of MiBS (in Mega Hertz)")
parser.add_argument("--BANDWIDTH_PBS", default=800,
                    help="The bandwidth of PBS (in Mega Hertz)")
# center frequency of base station
parser.add_argument("--CENTER_FREQUENCY_MABS", default=2000,
                    help="The center frequency of MaBS (in Mega Hertz)")
parser.add_argument("--CENTER_FREQUENCY_MIBS", default=2000,
                    help="The center frequency of MIBS (in Mega Hertz)")
parser.add_argument("--CENTER_FREQUENCY_PBS", default=28000,
                    help="The center frequency of PBS (in Mega Hertz)")
# each parameter of settings
parser.add_argument("--QOS_STANDARD", default=7,
                    help="The standard to evaluate the QoS of each user in terms of DL throughput (in dBm)")
parser.add_argument("--NOISE_SINR", default=-175,
                    help="The noise value in SINR (dBm/Hz)")
parser.add_argument("--RHO", default=1e-3,
                    help="The cost of unit power level")
parser.add_argument("--FAILUERE_COST", default=1e-2,
                    help="The failure cost for large gamma in R (13) equation")
# MAP
parser.add_argument("--X_MAX", default=2000,
                    help="The horizontal maximum distance of map (in meter)")
parser.add_argument("--Y_MAX", default=2000,
                    help="The vertical maximum distance of map (in meter)")
# Reward
parser.add_argument("--NU", default = 1000,
                    help="Weight of total revenue in (7)")
parser.add_argument("--IMPROVED_REWARD", default=2000,
                    help="Improved reward of system")
parser.add_argument("--REWARD_ASSOCIATION", default=10,
                    help="Reward value of system (base station association reward")
parser.add_argument("--REWARD_COLLISION", default=10,
                    help="Reward value of system (channel usage collision)")
parser.add_argument("--PENALTY_COLLISION", default=-4,
                    help="Penalty value of system (channel usage collision)")
parser.add_argument("--PENALTY_ASSOCIATION", default=-4,
                    help="Penalty value of system (base station association reward)")
h_params = parser.parse_args()
# 기지국, VUE 위치 (state)
Position_BS = np.zeros([61, 2], dtype=float)
Position_VUE = np.zeros([100, 2], dtype=float)
# VUE 속도
Velocity_VUE = np.zeros([100, 2], dtype=float)
# VUE들의 QoS, DL thr. (state)
QoS_VUE = np.zeros([100, 1], dtype=float)
DL_VUE = np.zeros([100, 1], dtype=float)
# 채널 상태 / # 미사용시 -1로 초기화, 아니면 VUE의 인덱스 값 입력. / # Channels_Usage[11]부터는 5열~29열은 -2로 초기화 및 미사용
Channel_Usage = np.zeros([61, 30], dtype=int) # Channel_Usage[0]~Channel_Usage[10]은 30열,
                                  # Channel_Usage[11]~Channel_Usage[60]은 5열.
# 채널 gain matrix / # 채널
Channel_Gain = np.zeros([61, 30], dtype=float) # Channel_Gain[0]~Channel_Gain[10]은 30열,
                                # Channel_Gain[11]~Channel_Gain[60]은 5열.
np.random.seed(h_params.SEED_VALUE)

"""
# 1. Initialize the environment.
"""
def Init_env():
    """
    1. 기지국과 차량 위치 초기화.
    """
    for i in range (1, 61):
        if (i >= 1):
            Position_BS[i][0] = np.random.randint(0, 2001)
            Position_BS[i][1] = np.random.randint(0, 2001)
    for i in range(0, h_params.NUM_VUE):
        Position_VUE[i][0] = np.random.randint(0, 2001)
        Position_VUE[i][1] = np.random.randint(0, 2001)
    Position_BS[0][0] = 1000
    Position_BS[0][1] = 1000

    """
    2. 차량의 이동 속력과 방향 정의
    """
    for i in range(0, h_params.NUM_VUE):
        Velocity_VUE[i][0] = np.random.randint(-12, 12)
        Velocity_VUE[i][1] = np.random.randint(-12, 12)

    """
    3. 채널의 사용과 채널 gain의 초기화
    """
    # 채널 사용정보(Usage) 초기화
    for i in range(0, 11):
        # MaBS, MiBS 채널 초기화
        for j in range(0, 30):
            Channel_Usage[i][j] = -1 # MaBS/MiBS 채널 미사용으로 초기화.
            Channel_Gain[i][j] = 0.0001 + np.random.random_sample() / 2 # 0 ~ 0.5

    for i in range(11, 61):
        # PBS 채널 초기화
        for j in range(0, 30):
            if j < 5:
                Channel_Usage[i][j] = -1 # PBS 채널 미사용으로 초기화.
                Channel_Gain[i][j] = 0.5 + np.random.random_sample() / 2 # 0.5 ~ 1
            elif j >=5 :
                Channel_Usage[i][j] = -2 # PBS의 채널에 해당사항 없음.
                Channel_Gain[i][j] = 0

    """
    4. QoS, DL Throughput 초기화
    """
    for i in range(0, h_params.NUM_VUE):
        QoS_VUE[i][0] = 0
        DL_VUE[i][0] = 0

    # Agent state : position, qos, dl
    state = np.hstack([Position_VUE, QoS_VUE, DL_VUE])
    return state # (100, 4)

"""
2. 차량 이동 및 거리 계산
"""
def Calc_Dist(i, j):
    BS = (Position_BS[i][0], Position_BS[i][1])
    VUE = (Position_VUE[j][0], Position_VUE[j][1])
    return distance.euclidean(BS, VUE)

def Move():
    for i in range(0, h_params.NUM_VUE):
        Position_VUE[i][0] += Velocity_VUE[i][0]
        # 좌측으로 이탈한 경우
        if(Position_VUE[i][0] < 0):
            Position_VUE[i][0] = 2000
            Position_VUE[i][1] = np.random.randint(0, 2001)
        # 우측으로 이탈한 경우
        if(Position_VUE[i][1] > 2000):
            Position_VUE[i][0] = 0
            Position_VUE[i][1] = np.random.randint(0, 2001)

"""
3. Vehicular network 상태 초기화
"""
def reset():
    Initial_state = Init_env()
    return Initial_state

def f(x):
    return np.int(x)
f2 = np.vectorize(f)
"""
4. Vehicular network 다음 상태 전이.
"""
def step(state, action):
    # 현재 vehicular network 상태와 agents들의 상태에 agents들의 액션을 반영하여 다음 상태로 전이.
    # agents 위치 (x, y), QoS, DL --> state of agents
    # action :
        # shape: 100 x 7
        # action domain 1: Which base station to associate (integer, bs index)
        # action domain 2: Number of channels to use (integer, ch index)
        # action domain 3: Which channels to use (list, ch list)

    # action scale to integer
    for i in range(0, h_params.NUM_VUE):
        for j in range(0, 7):
            action[i][j] = f2(action[i][j])

    step_reward = np.zeros([h_params.NUM_VUE, 1], dtype=float)
    penalty_wrong_base_station = np.zeros([h_params.NUM_VUE, 1], dtype=float) # cell association selection
    penalty_collision = np.zeros([h_params.NUM_VUE, 5], dtype=float) # channel selection (related to collision)

    # action에 대한 채널 사용 정보 갱신 및 충돌 penalty np 계산
    for i in range(0, h_params.NUM_VUE):
        # Cell association and resource allocation to vehicular network environment
        which_bs = action[i][0] # which base station to associate
        how_many_channels = action[i][1] # which channel(s) to associate

        """
        CARRIER AGGREGATION -> channel collision check and reward
        """
        channel_1 = action[i][2]
        channel_2 = action[i][3]
        channel_3 = action[i][4]
        channel_4 = action[i][5]
        channel_5 = action[i][6]

        # if MaBS, MiBS association
        if(which_bs >= 0 and which_bs < h_params.NUM_MABS+h_params.NUM_MIBS):
            for j in range(1, how_many_channels+1):
                # 채널 하나만 사용한다면
                if(j == 1):
                    # MaBS 연결시
                    if(which_bs == 0):
                        # 거리 내부라면
                        if(Calc_Dist(which_bs, i) <= h_params.CELL_MABS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            if(Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.PENALTY_COLLISION
                            else:
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i # 주파수 사용하는 유저 채널정보에 기록
                        # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j-1] = h_params.PENALTY_COLLISION

                    # MiBS 연결시
                    elif(which_bs > 0 and which_bs < h_params.NUM_MABS + h_params.NUM_MIBS):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MIBS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.PENALTY_COLLISION
                            else:
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록
                            # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j-1] = h_params.PENALTY_COLLISION


                # 두 개의 채널 사용시
                elif(j == 2):
                    # MaBS 연결시
                    if (which_bs == 0):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MABS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            # 첫 번째 채널
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if(Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록
                        # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j-2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j-1] = h_params.PENALTY_COLLISION

                    # MiBS 연결시
                    elif (which_bs > 0 and which_bs < h_params.NUM_MABS + h_params.NUM_MIBS):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MIBS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if (Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록
                            # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                # 세 개의 채널 사용시
                elif(j == 3):
                    # MaBS 연결시
                    if (which_bs == 0):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MABS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            # 첫 번째 채널
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-3] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if (Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 세 번째 채널
                            if (Channel_Usage[which_bs][channel_3] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_3] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록
                        # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                    # MiBS 연결시
                    elif (which_bs > 0 and which_bs < h_params.NUM_MABS + h_params.NUM_MIBS):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MIBS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-3] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if (Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 세 번째 채널
                            if (Channel_Usage[which_bs][channel_3] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_3] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록
                            # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                # 네 개의 채널 사용시
                elif(j == 4):
                    # MaBS 연결시
                    if (which_bs == 0):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MABS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            # 첫 번째 채널
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-4] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if (Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-3] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 세 번째 채널
                            if (Channel_Usage[which_bs][channel_3] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_3] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 네 번째 채널
                            if (Channel_Usage[which_bs][channel_4] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_4] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_4] = i  # 주파수 사용하는 유저 채널정보에 기록
                        # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                    # MiBS 연결시
                    elif (which_bs > 0 and which_bs < h_params.NUM_MABS + h_params.NUM_MIBS):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MIBS_COVERAGE):                            # 주파수 사용 확인 후 reward
                            # 첫 번째 채널
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-4] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if (Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-3] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 세 번째 채널
                            if (Channel_Usage[which_bs][channel_3] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_3] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 네 번째 채널
                            if (Channel_Usage[which_bs][channel_4] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_4] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_4] = i  # 주파수 사용하는 유저 채널정보에 기록
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                # 다섯개의 채널 사용시
                elif(j == 5):                    # MaBS 연결시
                    if (which_bs == 0):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MABS_COVERAGE):
                            # 주파수 사용 확인 후 reward
                            # 첫 번째 채널
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 5] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-5] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if (Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-4] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 세 번째 채널
                            if (Channel_Usage[which_bs][channel_3] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_3] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-3] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 네 번째 채널
                            if (Channel_Usage[which_bs][channel_4] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_4] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j-2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_4] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 네 번째 채널
                            if (Channel_Usage[which_bs][channel_5] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_5] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_5] = i  # 주파수 사용하는 유저 채널정보에 기록
                        # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j - 5] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                    # MiBS 연결시
                    elif (which_bs > 0 and which_bs < h_params.NUM_MABS + h_params.NUM_MIBS):
                        # 거리 내부라면
                        if (Calc_Dist(which_bs, i) <= h_params.CELL_MIBS_COVERAGE):                            # 주파수 사용 확인 후 reward
                            # 첫 번째 채널
                            if (Channel_Usage[which_bs][channel_1] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 5] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_1] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 5] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 두 번째 채널
                            if (Channel_Usage[which_bs][channel_2] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_2] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 4] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 세 번째 채널
                            if (Channel_Usage[which_bs][channel_3] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_3] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 3] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 네 번째 채널
                            if (Channel_Usage[which_bs][channel_4] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_4] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 2] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_4] = i  # 주파수 사용하는 유저 채널정보에 기록

                            # 네 번째 채널
                            if (Channel_Usage[which_bs][channel_5] != -1):
                                # 셀은 선택잘했지만 채널 선택 미흡
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                            elif (Channel_Usage[which_bs][channel_5] == -1):
                                # 셀 선택과 채널 선택 만족
                                penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                                penalty_collision[i][j - 1] = h_params.REWARD_COLLISION
                                Channel_Usage[which_bs][channel_5] = i  # 주파수 사용하는 유저 채널정보에 기록
                            # 거리 밖이라면
                        else:
                            # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                            penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                            penalty_collision[i][j - 5] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                            penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION


        # else if PBS association
        elif(which_bs >= h_params.NUM_MABS+h_params.NUM_MIBS and
             which_bs < h_params.NUM_MABS+h_params.NUM_MIBS+h_params.NUM_PBS):
            for j in range(1, how_many_channels+1):
                # 채널 하나만 사용한다면
                if(j == 1):
                    # 주파수 사용 확인 후 reward
                    # 첫 번째 채널
                    if (Channel_Usage[which_bs][channel_1] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_1] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 거리 밖이라면
                    else:
                        # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                elif (j == 2):
                    # 주파수 사용 확인 후 reward
                    # 첫 번째 채널
                    if (Channel_Usage[which_bs][channel_1] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_1] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 두 번째 채널
                    if (Channel_Usage[which_bs][channel_2] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_2] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록
                    # 거리 밖이라면
                    else:
                        # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                elif (j == 3):
                    # 주파수 사용 확인 후 reward
                    # 첫 번째 채널
                    if (Channel_Usage[which_bs][channel_1] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_1] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 3] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 두 번째 채널
                    if (Channel_Usage[which_bs][channel_2] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_2] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 세 번째 채널
                    if (Channel_Usage[which_bs][channel_3] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_3] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록
                    # 거리 밖이라면
                    else:
                        # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                        penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                elif (j == 4):
                    # 주파수 사용 확인 후 reward
                    # 첫 번째 채널
                    if (Channel_Usage[which_bs][channel_1] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_1] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 4] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 두 번째 채널
                    if (Channel_Usage[which_bs][channel_2] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_2] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 3] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 세 번째 채널
                    if (Channel_Usage[which_bs][channel_3] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_3] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록
                        
                    # 네 번째 채널
                    if (Channel_Usage[which_bs][channel_4] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_4] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_4] = i  # 주파수 사용하는 유저 채널정보에 기록
                    # 거리 밖이라면
                    else:
                        # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                        penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

                elif (j == 5):
                    # 주파수 사용 확인 후 reward
                    # 첫 번째 채널
                    if (Channel_Usage[which_bs][channel_1] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 5] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_1] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 5] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_1] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 두 번째 채널
                    if (Channel_Usage[which_bs][channel_2] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_2] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 4] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_2] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 세 번째 채널
                    if (Channel_Usage[which_bs][channel_3] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_3] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 3] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_3] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 네 번째 채널
                    if (Channel_Usage[which_bs][channel_4] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_4] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 2] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_4] = i  # 주파수 사용하는 유저 채널정보에 기록

                    # 다섯번 째 채널
                    if (Channel_Usage[which_bs][channel_5] != -1):
                        # 셀은 선택잘했지만 채널 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION
                    elif (Channel_Usage[which_bs][channel_5] == -1):
                        # 셀 선택과 채널 선택 만족
                        penalty_wrong_base_station[i][0] = h_params.REWARD_ASSOCIATION
                        penalty_collision[i][j - 1] = h_params.REWARD_COLLISION
                        Channel_Usage[which_bs][channel_5] = i  # 주파수 사용하는 유저 채널정보에 기록
                    # 거리 밖이라면
                    else:
                        # 다른 기지국들과 충돌과 더불어 셀 선택 미흡
                        penalty_wrong_base_station[i][0] = h_params.PENALTY_ASSOCIATION
                        penalty_collision[i][j - 5] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 4] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 3] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 2] = h_params.PENALTY_COLLISION
                        penalty_collision[i][j - 1] = h_params.PENALTY_COLLISION

    # 갱신된 채널 사용 정보를 바탕으로 SINR (3)
    # DL throughput (4)
    SINR = np.zeros([h_params.NUM_VUE, 1], dtype=float)
    Power_Cost = np.zeros([h_params.NUM_VUE, 1], dtype=float)
    for i in range(0, h_params.NUM_VUE):
        upper = 0
        down = 0
        cnt_channel = 0
        for j in range(5): # channel 사용 여부에 따라 분류
            if(penalty_wrong_base_station[i][0] == h_params.REWARD_ASSOCIATION and
            penalty_collision[i][j] == h_params.REWARD_COLLISION and action[i][j+2] != 0):
                if(action[i][0] == 0):
                    upper += Channel_Gain[action[i][0]][action[i][j+2]] * h_params.POWER_MABS
                    down += Channel_Gain[action[i][0]][action[i][j + 2]] * h_params.POWER_MABS
                    Power_Cost[i][0] += h_params.RHO * h_params.POWER_MABS
                    cnt_channel+=1
                elif(action[i][0] >=1 and action[i][0] <= h_params.NUM_MIBS):
                    upper += Channel_Gain[action[i][0]][action[i][j+2]] * h_params.POWER_MIBS
                    down += Channel_Gain[action[i][0]][action[i][j + 2]] * h_params.POWER_MIBS
                    Power_Cost[i][0] += h_params.RHO * h_params.POWER_MIBS
                    cnt_channel += 1
                elif(action[i][0] > h_params.NUM_MIBS):
                    upper += Channel_Gain[action[i][0]][action[i][j+2]] * h_params.POWER_PBS
                    down += Channel_Gain[action[i][0]][action[i][j + 2]] * h_params.POWER_PBS
                    Power_Cost[i][0] += h_params.RHO * h_params.POWER_PBS
                    cnt_channel += 1

            elif(penalty_wrong_base_station[i][0] == h_params.PENALTY_ASSOCIATION or
            penalty_collision[i][j] == h_params.PENALTY_COLLISION or action[i][j+2] != 0):
                if (action[i][0] == 0):
                    down += Channel_Gain[action[i][0]][action[i][j + 2]] * h_params.POWER_MABS
                elif (action[i][0] >= 1 and action[i][0] <= h_params.NUM_MIBS):
                    down += Channel_Gain[action[i][0]][action[i][j + 2]] * h_params.POWER_MIBS
                elif (action[i][0] > h_params.NUM_MIBS):
                    down += Channel_Gain[action[i][0]][action[i][j + 2]] * h_params.POWER_PBS

        if(action[i][0] == 0):
            # MABS 연결
            down += h_params.BANDWIDTH_MABS * h_params.NOISE_SINR
        elif(action[i][1] >=1 and action[i][0] <= h_params.NUM_MIBS):
            # MiBS 연결
            down += h_params.BANDWIDTH_MIBS * h_params.NOISE_SINR
        elif(action[i][0] > h_params.NUM_MIBS):
            # PBS 연결
            down += h_params.BANDWIDTH_PBS * h_params.NOISE_SINR

        if (down == 0):
            down += .01
        SINR[i][0] = upper / down

        if(1 + SINR[i][0] <= 0):
            SINR[i][0] = -.999
        if (action[i][0] == 0):
            DL_VUE[i][0] = cnt_channel * h_params.BANDWIDTH_MABS * math.log2(1 + SINR[i][0])
        elif (action[i][0] >= 1 and action[i][0] <= h_params.NUM_MIBS):
            DL_VUE[i][0] = cnt_channel * h_params.BANDWIDTH_MIBS * math.log2(1 + SINR[i][0])
        elif (action[i][0] > h_params.NUM_MIBS):
            DL_VUE[i][0] = cnt_channel * h_params.BANDWIDTH_PBS * math.log2(1 + SINR[i][0])

    # QoS, power-aware cost, total revenue 계산 (5~7)
    for i in range(h_params.NUM_VUE):
        if SINR[i][0] > h_params.QOS_STANDARD:
            QoS_VUE[i][0] = 1
        else:
            QoS_VUE[i][0] = -1

    for i in range(h_params.NUM_VUE):
        step_reward[i][0] = h_params.NU * DL_VUE[i][0] - Power_Cost[i][0] +\
                            penalty_collision[i][0] + \
                            penalty_collision[i][1] + \
                            penalty_collision[i][2] + \
                            penalty_collision[i][3] + \
                            penalty_collision[i][4] + \
                            penalty_wrong_base_station[i][0]

    # 다음 상태 정의
    state_ = np.hstack([Position_VUE, QoS_VUE, DL_VUE])

    # 차량 이동
    Move()

    # reward 계산, state_갱신
    return state_, step_reward