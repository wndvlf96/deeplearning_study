# Q-learning: 모델이 없는 강화학습
# 일단 각 state별 action들을 0으로 초기화
# 이후 각 state별 action에 대한 상수를 대입
# Q(s,a) = r + max(Q(s,a))
# decaying discount 적용이 안 되어있음
# exploit 만함 (exploration 안 함)

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vec):
    m = np.amax(vec)
    indices = np.nonzero(vec == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# 각 state의 action들을 0으로 초기화
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
rList = []

for i in range(num_episodes):
    # 반복문 돌때마다 게임 초기화
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        # 게임 한판 하기
        action = rargmax(Q[state, :])

        # done가 바뀌어서 반복문 탈출할수도
        # hall에 빠지거나 게임 끝났을 때
        new_state, reward, done,_ = env.step(action)

        Q[state,action] = reward + np.max(Q[new_state,:])
        
        # rAll: reward 더한 총값
        rAll += reward
        state = new_state
    
    # rList에 그간 num_episodes만큼 돈 rAll 값들 들어가게하기
    rList.append(rAll)

# 결과값 보기
print("성공율: "+str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("좌 하 우 상")
print(Q)
plt.bar(range(len(rList)), rList, color = 'blue')
plt.show()