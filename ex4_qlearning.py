# exlpoit(주로 가는곳 가기) exploration(새로운 시도)
# E-greedy (e를 고정으로 잡아서 확률적으로 explit 과 rxploraion 선택)
# decaying E-greedy (뒤로 갈수록 이 확률(e)이 줄어든다.)
# add random noise (이 확률 고정이 아니라 또 바뀜)(2 3번째의 선택을 할 때)
# 결국 action은 explit or exploration으로 선택하는 것!!!
# discounted reward: 미래의 액션은 작게 평가
# R = r + r a + r a ^ 2 + ...
# Q^(s,a) = r + a * max(Q^(s',a'))
# 이것으로 각 action에 대한 점수를 매기는 것!!!

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
# dicount 적용해서
a = 0.99
num_episodes = 2000
rList = []

for i in range(num_episodes):
    # 반복문 돌때마다 게임 초기화
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        # 게임 한판 하기
        # action을 선택하며
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n)/(i+1))

        # done가 바뀌어서 반복문 탈출할수도
        # hall에 빠지거나 게임 끝났을 때
        new_state, reward, done,_ = env.step(action)

        # discount값을 곱해주고 전 reward와 더하기
        Q[state,action] = reward + a * np.max(Q[new_state,:])
        
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