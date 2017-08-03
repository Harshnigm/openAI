import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter

LR=1e-3
env=gym.make('CartPole-v0')
#env.reset()
goal_steps=500
score_requirement=50
intial_games=10000


def random_games():
	for epi in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action=env.action_space.sample()
			observation,reward,done,info=env.step(action)
			if done:
				break

#random_games()

def initial_population():
	training_data=[]
	scores=[]
	accepted_scores=[]
	for _ in range(intial_games):
		score=0
		game_memory=[]
		prev_obs=[]
		env.reset()
		for _ in range(goal_steps):
			action=random.randrange(0,2)
			observation,reward,done,info=env.step(action)

			if len(prev_obs)>0:
				game_memory.append([prev_obs,action])
			prev_obs=observation
			score=score+reward
			if done:
				break
		if score>=score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1]==1:
					training_data.append([data[0],[0,1]])
				else:
					training_data.append([data[0],[1,0]])
		env.reset()

		scores.append(score)

	training_data_save=np.array(training_data)
	np.save('saved.npy',training_data_save)

	print('average accepted scores-',mean(accepted_scores))
	print('median accepted scores-',median(accepted_scores))
	print('frequency accepted scores-',len(accepted_scores))
	return training_data




def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model



def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=1, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data=initial_population()
model=train_model(training_data)


scores = []
choices = []
for each_game in range(1):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        #env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            print(prev_obs.reshape(-1,len(prev_obs),1))
        choices.append(action)
        #print(np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))))
                
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)

model.save('200.model')
