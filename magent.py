import tensorflow as tf
import numpy as np
import math
from model.model_MADDPG import MADDPG
from util.replay_buffer import ReplayBuffer
import env
import csv
import pandas as pd

gpu_fraction = 0.6
num_agents = 100
Episode = 10000

agent_list = []
agent_target_list = []

agent_actor_target_init_list = []
agent_actor_target_update_list = []

agent_critic_target_init_list = []
agent_critic_target_update_list = []

agent_action_list = []
memory = []
def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

    return target_init, target_update

for agents in range(0, num_agents):
    agent = MADDPG('agent'+str(agents))
    agent_target = MADDPG('agent'+str(agents)+'_target')
    agent_list.append(agent)
    agent_target_list.append(agent_target)

saver = tf.train.Saver()

for init in range(0, num_agents):
    agent_actor_target_init, agent_actor_target_update = create_init_update('agent'+str(init)+'_actor', 'agent'+str(init)+'_target_actor')
    agent_critic_target_init, agent_critic_target_update = create_init_update('agent' + str(init) + '_critic',
                                                                            'agent' + str(init) + '_target_critic')
    agent_actor_target_init_list.append(agent_actor_target_init)
    agent_actor_target_update_list.append(agent_actor_target_update)
    agent_critic_target_init_list.append(agent_critic_target_init)
    agent_critic_target_update_list.append(agent_critic_target_update)

def get_agents_action(o_n, sess, noise_rate=0.):
    for agents in range(0, num_agents):
        agent_action = agent_list[agents].action(state=[o_n[agents]], sess=sess) + np.random.randn(7)*noise_rate
        agent_action_list.append(agent_action)
    mat = np.asarray(agent_action_list)
    mat = np.squeeze(mat)
    agent_action_list.clear()
    #print(mat.shape)
    return mat

def action_scaling(actions, number):
    # action scaling - 1. base station selection scaling
    for i in range(0, number):
        for j in range(0, 7):
            if(math.isnan(actions[i][j])):
                actions[i][j] = -0.1

    if(math.isnan(actions[0][0])):
        actions[0][0] = -0.1
    min_bs = max_bs = actions[0][0]
    for j in range(0, number):
        if (math.isnan(actions[j][0])):
            actions[j][0] = -0.1
        if actions[j][0] > max_bs:
            max_bs = actions[j][0]
        if actions[j][0] < min_bs:
            min_bs = actions[j][0]
    if min_bs < 0:
        min_bs = -min_bs
    bs_max = min_bs + max_bs  # 0~x범위로 이동 후 x값.
    if(bs_max == 0):
        bs_max = 0.1
    for j in range(0, number):
        actions[j][0] += min_bs
        actions[j][0] = np.rint(actions[j][0] / bs_max * 60)
    # print("1. 어떤 기지국 사용할지")
    # print(actions)

    # action scaling - 2. number of channels to use

    if (math.isnan(actions[0][1])):
        actions[0][1] = -0.1
    min_chan_num = max_chan_num = actions[0][1]
    for j in range(number):
        if (math.isnan(actions[j][1])):
           actions[j][1] = -0.1
        if actions[j][1] > max_chan_num:
            max_chan_num = actions[j][1]
        if actions[j][1] < min_chan_num:
            min_chan_num = actions[j][1]
    if min_chan_num < 0:
        min_chan_num = -min_chan_num
    chan_max = min_chan_num + max_chan_num  # 0~x범위로 이동 후 x값.
    if(chan_max == 0):
        chan_max = 0.1
    for j in range(number):
        if(math.isnan(actions[j][1])):
            actions[j][1] = -0.1
        actions[j][1] += min_chan_num
        actions[j][1] = np.rint(actions[j][1] / chan_max * 5)
    # print("2. 채널 몇 개 사용할지")
    # print(actions)

    # action scaling - 3. what channels
    for j in range(number):
        if (math.isnan(actions[j][2])):
           actions[j][2] = -0.1
        min_chan_sel = max_chan_sel = actions[j][2]
        for k in range(2, 7):
            if (math.isnan(actions[j][k])):
               actions[j][k] = -0.1
            if actions[j][k] < min_chan_sel:
                min_chan_sel = actions[j][k]
            if actions[j][k] > max_chan_sel:
                max_chan_sel = actions[j][k]
        if min_chan_sel < 0:
            min_chan_sel = -min_chan_sel
        tomax_chan_sel = min_chan_sel + max_chan_sel
        if(tomax_chan_sel == 0):
            tomax_chan_sel = 0.1
        for k in range(2, 7):
            actions[j][k] += min_chan_sel
            if (actions[j][0] < 11):  # 30까지 가능
                actions[j][k] = np.rint(actions[j][k] / tomax_chan_sel * 29)
            else:
                actions[j][k] = np.rint(actions[j][k] / tomax_chan_sel * 4)

        if(math.isnan(actions[j][1])):
            actions[j][1] = -0.1
        for k in range(2 + int(actions[j][1]), 7):
            #if (math.isnan(actions[j][k])):
            #    actions[j][k] = 1.
            actions[j][k] = 1.
    # print("3. 최종 actions")
    # print(actions)
    actions = actions.astype(np.int)
    #print("정수형 actions")
    #print(actions)
    return actions

# For each agent
def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch = agent_memory.Sample(32)
    #print("total_obs_batch")
    #print(total_obs_batch)

    # 1. 전체 상태의 배치 (32, 100, 4)
    tot_obs_batch = np.asarray(total_obs_batch)
    #print("total_obs_batch_np:")
    #print(tot_obs_batch.shape)
    #print(tot_obs_batch)

    #print("total_act_batch")
    #print(total_act_batch)

    # 2, 전체 액션의 배치 (32, 100, 7)
    tot_act_batch = np.asarray(total_act_batch)
    #np.delete(tot_act_batch, 0, axis=1)
    #print("total_action_batch_np:")
    #print(tot_act_batch.shape)
    #print(tot_act_batch)

    # 3. 전체 리워드 (32,) -> (32, 100?)
    reward_batch = np.asarray(rew_batch)
    #print("reward_batch_np")
    #print(reward_batch.shape)
    #print(reward_batch)

    # 4. 다음 전체 상태의 배치
    tot_next_obs_batch = np.asarray(total_next_obs_batch)
    #print("tot_next_obs_batch_np:")
    #print(tot_next_obs_batch.shape)
    #print(tot_next_obs_batch)

    # 현재 액터의 액션 배치
    act_batch = tot_act_batch[:, 0, :]
    cur_act_batch = np.asarray(act_batch)
    #print("cur_act_batch:")
    #print(cur_act_batch.shape) # 32, 7
    #print(cur_act_batch)

    other_act_batch = tot_act_batch[:, 1:num_agents, :]
    cur_other_act_batch = np.asarray(other_act_batch)
    #print("다른 act batch?")
    #print(cur_other_act_batch.shape)
    #print(cur_other_act_batch)
    cur_other_act_batch = cur_other_act_batch.reshape((32, 99*7))

    cur_obs_batch = tot_obs_batch[:, 0, :]
    #print("이 agent의 obs_batch")
    cur_obs_batch = np.asarray(cur_obs_batch)
    cur_obs_batch.reshape((32, 4))
    #print(cur_obs_batch)

    next_obs_batch = np.asarray(tot_next_obs_batch[:, :, :]) # 32, 100, 4
    next_cur_obs_batch = np.asarray(tot_next_obs_batch[:, 0, :]) # 32, 1, 4
    next_cur_obs_batch = next_cur_obs_batch.reshape((32, 4))

    cur_agent_next_action_batch = agent_ddpg.action(next_cur_obs_batch.squeeze(), sess)
    #print("다음액션배치 ")
    #print(cur_agent_next_action_batch.shape)
    #print(cur_agent_next_action_batch)

    cur_agent_next_action_batch = action_scaling(cur_agent_next_action_batch, 32)

    #print("!!현재에이전트의 다음 액션 배치 shape:(32, 7가 되어야)")
    #print(cur_agent_next_action_batch.shape)

    next_other_action = other_actors[0].action(tot_obs_batch[:, 1, :], sess)
    #print("첫 다른 actor의 action 모양")
    #print(next_other_action.shape)
    n_other_action = np.asarray(next_other_action)

    for i in range(1, num_agents-1):
        #print(str(i+1)+"번째 에이전트의 액션 vstacking..")
        n_other_action = np.vstack([n_other_action, other_actors[i].action(tot_obs_batch[:, i+1, :], sess)])

        #print("현재 next_other_action shape은?")
        #print(n_other_action.shape)
    # scaling
    n_other_action = action_scaling(n_other_action, 32*(num_agents-1))
    # reshaping
    #n_other_action = n_other_action.reshape((32, -1, 7))

    #print("최종 other action?")
    #print(n_other_action.shape)
    #print(n_other_action)
    n_other_action = n_other_action.reshape((32, 7*99))
    # Q 에 넣기 위해 n_other_action 을 -> 3168, 7 -> (32, 7*99)으로


    #print("!!다른 에이전트들의 다음 액션 배치 shape:(32, 7*99가 되어야)")
    #print(n_other_action)

    # ravel to feed the variables to corresponding placeholders.
    next_obs_batch = next_obs_batch.reshape((-1, 4))

    #next_other_action = next_other_action_bef
    #next_other_action = np.hstack([other_actors[0:].action(next_other_actors_o[0:], sess)])
    #target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),
                                                                     #other_action=n_other_action_np, sess=sess)
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_cur_obs_batch, #32 x 4
                                                                     action=cur_agent_next_action_batch, # 32 x 7
                                                                     other_action=n_other_action, sess=sess) # 32 x 7*99
    agent_ddpg.train_actor(state=cur_obs_batch, other_action=cur_other_act_batch, sess=sess)
    agent_ddpg.train_critic(state=cur_obs_batch, action=cur_act_batch, other_action=cur_other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])

if __name__ == '__main__':
    o_n = env.reset()
    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(num_agents)]
    agent_reward_op = [tf.summary.scalar('agent'+str(i)+'_reward', agent_reward_v[i]) for i in range(num_agents)]

    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(num_agents)]
    agent_a1_op = [tf.summary.scalar('agent'+str(i)+'_action_1', agent_a1[i]) for i in range(num_agents)]

    agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(num_agents)]
    agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a1[i]) for i in range(num_agents)]

    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(num_agents)]
    reward_100_op = [tf.summary.scalar('agent'+str(i)+'_reward_100_mean', reward_100[i]) for i in range(num_agents)]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run([agent_actor_target_init_list[:], agent_critic_target_init_list[:]])

    summary_writer = tf.summary.FileWriter('./VUE_summary', graph=tf.get_default_graph())

    for i in range(num_agents):
        mem = ReplayBuffer(10000)
        memory.append(mem)

    # for every 100 step, check the rewards
    reward_100_list = np.zeros([100, 1], dtype=float)
    sum_r = 0.
    for i in range(1, Episode+1):
        print(str(i)+"번째 에피소드 시작..")
        if i % 100 == 0:
            print(str(i)+"번째 에피소드. 환경 리셋.(100 배수)")
            o_n = env.reset()
            for agent_index in range(num_agents):
                summary_writer.add_summary(sess.run(reward_100_op[agent_index], {reward_100[agent_index]: np.mean(reward_100_list[agent_index])}), i//1000)
        #print("현재 상태")
        #print(o_n) # 100, 4

        # action get
        print("step 함수호출")
        actions = get_agents_action(o_n, sess, noise_rate=0.2)  # 100, 7
        actions = action_scaling(actions, num_agents)
        o_n_next, r_n = env.step(o_n, actions)
        #print("현재 리워드")
        #print(r_n)

        for agent_index in range(num_agents):
            np.hstack([reward_100_list[agent_index], r_n[agent_index]])
            #np.concatenate(reward_100_list[agent_index],r_n[agent_index])
        reward_100_list = np.squeeze(reward_100_list)
        actions = np.squeeze(actions)
        r_n = np.squeeze(r_n)
        o_n_next = np.squeeze(o_n_next)
        o_n = np.squeeze(o_n)

        print("각 에이전트들마다 메모리 생성 및 저장")
        for agent_index in range(num_agents):
            memory[agent_index].add(np.vstack([o_n[agent_index], o_n[agent_index+1:], o_n[0:agent_index]]),
                                    np.vstack([actions[agent_index], actions[agent_index+1:], actions[0:agent_index]]),
                                    r_n[agent_index],
                                    np.vstack([o_n_next[agent_index], o_n_next[agent_index+1:], o_n_next[0:agent_index]]))


        print(str(i) + "번째 학습 시작")
        for agent_index in range(0, num_agents):
            agent_target_element = agent_target_list[agent_index]
            rest_target_list = []
            for k in range(0, num_agents):
                if k != agent_index:
                    rest_target_list.append(agent_target_list[k])

            #train_agent(agent_list[agent_index], agent_target_list[agent_index], memory[agent_index], agent_actor_target_update_list[agent_index],
                        #agent_critic_target_update_list[agent_index], sess, agent_target_list[~agent_index])
            train_agent(agent_list[agent_index], agent_target_list[agent_index], memory[agent_index],
                        agent_actor_target_update_list[agent_index],
                        agent_critic_target_update_list[agent_index], sess, rest_target_list)
            # 위 train결과로 reward, QoS, DL throughput return하도록하고 아래에서 출력해 결과 확인.
            print(str(i)+"번째 에피소드의 " + str(agent_index+1)+"번째 에이전트 학습 완료")

        if i % 10 == 0:
            print("*" * 50)
            print("*" * 50)

            print(str(i) + "번째 에피소드")

            print("1. " + str(i) + "번째 학습 후 각 에이전트의 상태")
            print(o_n)  # (100, 4)
            df_states = pd.DataFrame(o_n)
            with open("Agents_States.csv", 'a+') as STATES:
                df_states.to_csv(STATES, header=None, index=False, encoding='utf-8')

            print("2. " + str(i) + "번째 학습 후 각 에이전트의 액션")
            print(actions)  # (100, 7)
            df_actions = pd.DataFrame(actions)
            with open("Agents_Actions.csv", 'a+') as ACTIONS:
                df_actions.to_csv(ACTIONS, header=None, index=False, encoding='utf-8')

            print("3. " + str(i) + "번째 학습 후 각 에이전트의 리워드")
            print(r_n)  # (100, 1)
            df_rewards = pd.DataFrame(r_n)
            with open("Agents_Rewards.csv", 'a+') as REWARDS:
                df_rewards.to_csv(REWARDS, header=None, index=False, encoding='utf-8')

            print("4. " + str(i) + "번째 학습 후 에이전트들의 리워드 평균")
            print(np.mean(r_n))  # scalar


            print("*" * 50)
            print("*" * 50)

            for agent_index in range(num_agents):
                summary_writer.add_summary(sess.run(agent_reward_op[agent_index], {agent_reward_v[agent_index]: r_n[agent_index]}), i)
                summary_writer.add_summary(sess.run(agent_a1_op[agent_index], {agent_a1[agent_index]: actions[agent_index][0]}), i)
                summary_writer.add_summary(sess.run(agent_a2_op[agent_index], {agent_a2[agent_index]: actions[agent_index][1]}), i)
                summary_writer.add_summary(sess.run(reward_100_op[agent_index], {reward_100[agent_index]: np.mean(reward_100_list[agent_index])}), i)
            # print('Total episodic reward : ' + str(sum_r / i))
            saver.save(sess, './100VUE_weight/' + str(i) + '.ckpt')

        o_n = o_n_next


    print("학습 종료")
    sess.close()