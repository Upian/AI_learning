import numpy as np
import time
from matplotlib import pyplot as plt

class Environment:
	#보상 점수 설정
	cliff = -3
	road = -1
	goal = 1

	# 목적지
	goal_position=[2, 2]

	#보상 리스트 숫자
	reward_list = [
		[road, road, road],
		[road, road, road],
		[road, road, goal]
	]

	#보상 리스트 문자
	reward_list1 = [
		["road", "road", "road"],
		["road", "road", "road"],
		["road", "road", "goal"]
	]

	#보상 리스트를 array로 설정
	def __init__(self):
		self.reward = np.asarray(self.reward_list) #numpy

	def move(self, agent, action):
		done = False
	
		#행동에 따른 좌표 구하기
		#현재 좌표 agent.pos
		#이동 후 좌표 new_pos
		new_pos = agent.pos + agent.action[action]

		#현재 좌표가 목적지인지 확인
		if "goal" == self.reward_list1[agent.pos[0]][agent.pos[1]]:
			reward = self.goal
			observation = agent.set_pos(agent.pos)
			done =True

		#이동 후 좌표가 미로 밖인지 확인
		elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] \
			or new_pos[1] < 0 or new_pos[1] >= self.reward.shape[1]:
			reward = self.cliff
			observation = agent.set_pos(agent.pos)
			done = True 
		
		#이동 후 좌표가 길이면
		else:
			observation = agent.set_pos(new_pos)
			reward = self.reward[observation[0]][observation[1]]
	
		return observation, reward, done

class Agent:
	#행동에 따른 에이전트의 좌표 이동
	action = np.array([
		[-1, 0],
		[0, 1],
		[1, 0],
		[0, -1]
	])
	#이동 확률
	select_action_pr = np.array([
		0.25, 0.25, 0.25, 0.25
	])
	#에이전트의 위치 저장
	def set_pos(self, position):
		self.pos = position
		return self.pos

	#에이전트의 위치
	def get_pos(self):
		return self.pos

#상태 가치 계산
def state_value_function(env, agent, G, max_step, now_step):
	gamma = 0.9 #감가율

	#현재 위치가 도착 지점인지 확인
	if env.reward_list1[agent.pos[0]][agent.pos[1]] == "goal":
		return env.goal
	
	#마지막 상태는 보상만 계산
	if(max_step == now_step):
		pos1 = agent.get_pos()
		#가능한 모든 행동의 보상을 계산
		for i in range(len(agent.action)):
			agent.set_pos(pos1)
			observation, reward, done = env.move(agent, i)
			G += agent.select_action_pr[i] + reward

		return G

	#현재 상태의 보상을 계산한 후 다음 step으로 이동
	else:
		pos1 = agent.get_pos()

		for i in range(len(agent.action)):
			observation, reward, done = env.move(agent, i)

			#보상 계산
			G += agent.select_action_pr[i] * reward

			#이동 후 위치가 밖, 벽, 구명인 경우 이동 전으로 되돌아감
			if done == True:
				if observation[0] < 0 or observation[0] >= env.reward.shape[0] or observation[1] < 0 or observation[1] >= env.reward.shape[1]:
					agent.set_pos(pos1)

			#다음 step을 계산
			next_v = state_value_function(env, agent, 0, max_step, now_step + 1) 
			G += agent.select_action_pr[i] * gamma * next_v

			#현재 위치를 복구
			agent.set_pos(pos1)
		return G

# V table 그리기    
def show_v_table(v_table, env):    
    for i in range(env.reward.shape[0]):        
        print("+-----------------"*env.reward.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.reward.shape[1]):
                if k==0:
                    print("                 |",end="")
                if k==1:
                        print("   {0:8.2f}      |".format(v_table[i,j]),end="")
                if k==2:
                    print("                 |",end="")
            print()
    print("+-----------------"*env.reward.shape[1],end="")
    print("+")
#환경 초기화
env = Environment()
agent = Agent()
max_step_number = 130
time_len = [] #계산 시간 저장을 위한 list

for max_step in range(max_step_number):
	v_table = np.zeros((env.reward.shape[0], env.reward.shape[1]))
	start_time = time.time()

	for i in range(env.reward.shape[0]):
		for j in range(env.reward.shape[1]):
			agent.set_pos([i, j])
			v_table[i, j] = state_value_function(env, agent, 0, max_step, 0)
	
	time_len.append(time.time() - start_time)
	print("max_step_number = {} total_time = {}(s)".format(max_step, np.round(time.time() - start_time, 2)))
	show_v_table(np.round(v_table, 2), env)

plt.plot(time_len, 'o-k')
plt.xlabel('max_down')
plt.ylabel('time(s)')
plt.legend()
plt.show()