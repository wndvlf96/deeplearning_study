# deterministic : 항상 정해져있음(오른쪽으로 가면 한칸 옆) -> is_slippery를 False로 선택
# stochastic(non-deterministic) : 오른쪽 눌러도 안 움직이기 아래로 움직이기 할수도
# Q로 했을 때 랜덤으로 한 것 보다 안 좋음!!!
# Q의 경험대로 한다해도 그닥 안 좋음
# 그래서: 멘토를 많이 둔다!
# Learning rate를 둬서 Q를 조금만(러닝레이트만큼만) 듣기!
# Q(s,a) = (1- r)Q(s,a) + r * max(Q(s',a'))
# 위가 최종적 Q알고리즘 수식!!!
# Q^ 이 Q에 많이 하면 수렴하게 된다!!!

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

# 이거 v0는 nondeterministic 환경입니다!!!
env = gym.make('FrozenLake-v0')

# 각 state의 action들을 0으로 초기화
Q = np.zeros([env.observation_space.n, env.action_space.n])
# learning rate를 적용해서 non_deterministic에서도 정확도를 약간 높여보자!
# 클수록 빠르게 학습
learning_rate = 0.85
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

        # 러닝레이트를 적용해서 한것!
        Q[state,action] = (1 - learning_rate)*Q[state, action] + learning_rate*(reward + a * np.max(Q[new_state,:]))
        
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