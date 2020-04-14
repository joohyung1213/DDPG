import numpy as np
import random
import math
from argparse import ArgumentParser
# ===============================================================================
# Number of VUE: 100
# Number of MaBS, MiBS, and PBS: 1, 10, 50

# < PBS >
# mmWave pathloss model: mmMAGIC (UMi scenario, mmMAGIC UMi-street Canyon LOS)
#                        19.2 log(d) + 32.9 + 20.8log(f_c)
# V band(57 - 66GHz), fc: 60GHz, W = 500MHz (20 channels)


# < MaBS/MiBS >
# MaBS/MiBS PL: 34 + 40log(d)
# fc: 2.5GHz, W = 10MHz (20 channels)

# cell coverage(MaBS, MiBS, and PBS): 3000, 500, 100m
# Transmit power of MaBS, MiBS, and PBS: 40dBm, 35dBm, and 55dBm.
# ===============================================================================
# ============================ Hyperparameter ===================================
random.seed(1)

# Power of BS (dBm)
Power_MaBS = 40
Power_MiBS = 35
Power_PBS = 55

# Coverage region of BS (meter)
Coverage_MaBS = 3000
Coverage_MiBS = 500
Coverage_PBS = 100

# Channel number of BS
Channels_MAIBS_NUM = 20
Channels_PBS_NUM = 20

# Center frequency (GHz)
fc_PBS = 60
fc_MAIBS = 2.5

# Bandwidth (MHz)
W_PBS = 500
W_MAIBS = 10

# Number of entity.
MaBS_NUM = 1
MiBS_NUM = 10
PBS_NUM = 50
BS_NUM = MaBS_NUM + MiBS_NUM + PBS_NUM
VUE_NUM = 100

# Position (meter)
BS_Position = []
VUE_Position = []

# MAP
X_MAX = 2000
Y_MAX = 2000

# Vehicle movement (What direction and what meters to go for each step)
Movement = []
BS = []

# Reward
ImproveRwd = 2000.
Penalty = -500.
# ===============================================================================
def Init_Environment():
    # Position initialization of BSs.
    BS_Position.append([1000, 1000])
    for MiBSPos in range(MiBS_NUM):
        x = random.randint(0, X_MAX)
        y = random.randint(0, Y_MAX)
        BS_Position.append([x, y])

    for PBSPos in range(PBS_NUM):
        x = random.randint(0, X_MAX)
        y = random.randint(0, Y_MAX)
        BS_Position.append([x, y])

    # Position initialization of VUEs.
    for VUE in range(VUE_NUM):
        x = random.randint(0, X_MAX)
        y = random.randint(0, Y_MAX)
        VUE_Position.append([x, y])

        for VUE_Velo in range(VUE_NUM):
            x = random.randint(10, 15)
            y = random.randint(10, 15)
            if VUE_Velo % 4 == 0:
                Movement.append([x, y])
            elif VUE_Velo % 4 == 1:
                Movement.append([x, -y])
            elif VUE_Velo % 4 == 2:
                Movement.append([-x, -y])
            else:
                Movement.append([-x, y])
    Possi_BS = PossibleBS(VUE_Position, BS_Position)
    state = []
    PL = np.zeros([VUE_NUM, 1])
    for VUE in range(VUE_NUM):
        Pos_BS = Possi_BS[VUE]
        BS_idx = random.sample(Pos_BS, 1)
        BS = BS_idx[0]
        if BS == 0:
            channel = random.randint(0, Channels_MAIBS_NUM)
            state.append([BS, channel])
        elif BS >= 1 and BS <= 10:
            channel = random.randint(0, Channels_MAIBS_NUM)
            state.append([BS, channel])
        else:
            channel = random.randint(0, Channels_PBS_NUM)
            state.append([BS, channel])
    state = np.hstack((state, PL)) # shape of state = (VUE_NUM, 3) : BS, channel, PL
    return state
# ===============================================================================
def reset():
    Init_state = Init_Environment()
    return Init_state
# ===============================================================================
def PossibleBS(VUE_Position, BS_Position):
    Possi_BS = []
    for VUE in range(VUE_NUM):
        ForthisBS = []
        for BS in range(PBS_NUM+MiBS_NUM+MaBS_NUM):
            x1 = VUE_Position[VUE][0]
            y1 = VUE_Position[VUE][1]
            x2 = BS_Position[BS][0]
            y2 = BS_Position[BS][1]
            dis = math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2))
            if BS == 0:                # MaBS
                if dis < Coverage_MaBS:
                    ForthisBS.append(BS)
            elif BS >= 1 and BS <= 10: # MiBS
                if dis < Coverage_MiBS:
                    ForthisBS.append(BS)
            else:                      # PBS
                if dis < Coverage_PBS:
                    ForthisBS.append(BS)
        Possi_BS.append(ForthisBS)
    return Possi_BS # VUE마다 연결가능한 BS목록 리턴.

# 개별 VUE 인덱스마다 연결가능 BS 목록 return.
def PossibleBS_Index(VUE_idx, BS_Position):
    Possi_BS = []
    for BS in range(BS_NUM):
        if BS == 0 and getDistance(VUE_idx, BS) < Coverage_MaBS:
            Possi_BS.append(BS)
        elif BS >= 1 and BS <= MiBS_NUM and getDistance(VUE_idx, BS) < Coverage_MiBS:
            Possi_BS.append(BS)
        elif BS > MiBS_NUM and getDistance(VUE_idx, BS) < Coverage_PBS:
            Possi_BS.append(BS)
    return Possi_BS

def getDistance(VUE, BS):
    # input: VUE index, BS index
    x1 = VUE_Position[VUE][0]
    y1 = VUE_Position[VUE][1]
    x2 = BS_Position[BS][0]
    y2 = BS_Position[BS][1]
    dis = math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2))
    return dis
# ===============================================================================
"""
MaBS_Position, MiBS_Position, PBS_Position, VUE_Position
Coverage_MaBS, Coverage_MiBS, Coverage_PBS
"""
# ===============================================================================
def step(state, action):
    # state: BS, channel --> whole state.                                           (BS, channel, PL) --> (None, 3)
    # action: what BS to associate, what channel to utilize. --> whole actions.     (BS, channel) -->     (None, 2)
    # reward: compare before/after state
    # VUE position change
    # resource utilization state change
    # reward return

    """
    # mmWave pathloss model: mmMAGIC (UMi scenario, mmMAGIC UMi-street Canyon LOS)
    #                        19.2 log(d) + 32.9 + 20.8log(f_c)
    # V band(57 - 66GHz), fc: 60GHz, W = 500MHz (20 channels)

    # < MaBS/MiBS >
    # MaBS/MiBS PL: 34 + 40log(d)
    # fc: 2.5GHz, W = 10MHz (20 channels)
    """
    # 이전 state와 PL비교
    # PL 계산 후 이전 PL값보다 작아졌을 때 BS, channel 선택 더 잘한 것으로.
    state_ = np.zeros([VUE_NUM, 3])

    # state_ PL 계산
    # 각 agent의 상태 shape: (1, 3)
    #action = [np.int(x) for x in action] # to int
    action = np.asarray(action)
    action = action.astype(int)
    #print(action.shape) --> (100, 1, 2)
    action = np.squeeze(action)
    # print(action.shape) --> (100, 2)
    for i in range(0, VUE_NUM):
        if action[i][0] < 0:
            action[i][0] = 0
            state_[i][0] = action[i][0]
        elif action[i][0] >= BS_NUM:
            action[i][0] = BS_NUM-1
            state_[i][0] = BS_NUM-1
        else:
            state_[i][0] = action[i][0]

    for i in range(0, VUE_NUM):
        if action[i][1] < 0:
            action[i][1] = 0
            state_[i][1] = 0
        elif action[i][1] >= Channels_PBS_NUM:
            action[i][1] = Channels_PBS_NUM - 1
            state_[i][1] = Channels_PBS_NUM - 1
        else:
            state_[i][1] = action[i][1]

    #state_[:, 0:2] = action.copy()
    for VUE in range(VUE_NUM):
        if state_[VUE][0] <= MiBS_NUM:
            # Follow MaBS/MiBS PL model
            # 34 + 40log(d)
            dis = getDistance(VUE, (int)(state_[VUE][0]))
            PL = 34 + 40*math.log10(dis)
            state_[VUE][2] = PL
        else:
            # Follow PBS PL model
            # 19.2 log(d) + 32.9 + 20.8log(f_c)
            dis = getDistance(VUE, (int)(state_[VUE][0]))
            PL = 19.2*math.log10(dis) + 32.9 + 20.8*math.log10(fc_PBS)
            state_[VUE][2] = PL

    # 각 reward 계산
    # state / state_의 PL 비교
    Reward = np.zeros([VUE_NUM, 1])
    for VUE in range(VUE_NUM):
        if state[VUE][2] == 0:
            # for initial state:
            Reward[VUE] = ImproveRwd
        elif state[VUE][2] > state_[VUE][2]:
            # reward
            Reward[VUE] = ImproveRwd
        else:
            Reward[VUE] = Penalty

    for VUE in range(VUE_NUM):
        VUE_Position[VUE][0] += Movement[VUE][0]
        VUE_Position[VUE][1] += Movement[VUE][1]
        if VUE_Position[VUE][0] < 0:
            VUE_Position[VUE][0] += 2000
        elif VUE_Position[VUE][0] > 2000:
            VUE_Position[VUE][0] -= 2000
        elif VUE_Position[VUE][1] < 0:
            VUE_Position[VUE][1] += 2000
        elif VUE_Position[VUE][1] > 2000:
            VUE_Position[VUE][1] -= 2000

    #Possi_BS = PossibleBS(VUE_Position, BS_Position) # 다음 상태에서 VUE들이 선택 가능한 BS목록(for action)
    return state_, Reward
# ===============================================================================