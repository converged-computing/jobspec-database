import numpy as np
import pandas as pd
from tpg.trainer import Trainer
from tpg.agent import Agent
import multiprocessing as mp
import time
import ast, pickle
import zlib
from pathlib import Path
from tpg.utils import getLearners, getTeams, learnerInstructionStats, actionInstructionStats, pathDepths


#values that can be modified for testing 
MAX_STEPS_G = 500 #max values we want for training starts with 1 (so subtract one)
GENERATIONS = 2000
EXTRA_TIME_STEPS  = 50 #number of wanted generated values 
STARTING_STEP = 0 #starting step
DATA_DIVISION = 0.5

data1 = pd.read_csv("./input.csv", header = 0)

MAX_PITCHES = data1['pitch'].apply(ast.literal_eval).apply(len).max() #max pitches played in a step for whole piece


def root_mean_squared_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("The length of actual and predicted lists must be the same.")
    
    sum_squared_error = 0
    for a, p in zip(actual, predicted):
        sum_squared_error += (a - p) ** 2
    rmse = (sum_squared_error / len(actual)) ** 0.5
    return rmse

#environment
class TimeSeriesEnvironment:
    def __init__(self, max_steps = MAX_STEPS_G, current_step_g=STARTING_STEP):

        self.max_generated_steps = max_steps
        self.current_step = current_step_g

        #starts off at 0, getting the 0th row in the dataset
        self.last_state = self._get_state()

    def reset(self, episodenum, window_size):
        # Resets to the starting step we want
        
        self.current_step = episodenum * window_size 

        #0 to 20, subtract 1  
        self.max_generated_steps = (self.current_step + window_size)
        self.last_state = self._get_state() 

        return self.last_state


#get predicted step and compare with the actual value
    
#first step: 0, last step:  19. len: 20
    def step(self, action_type):

        #reset values 
        done = False
        reward = 0

        #get the state we are on
        self.last_state = self._get_state()
        #intuition is that if its the first atomic action, it will directly use the value, else (second atomic action), it will apply the change to the previous
        if action_type[0] == 0:
            modified_offset = action_type[1][0] #+ self.last_state[0]
            modified_duration = action_type[1][1] #+ self.last_state[1]
            modified_pitch_array = action_type[1][2:] #+ self.last_state[2]
            
        else: 
            modified_offset = action_type[1][0] + self.last_state[0]
            modified_duration = action_type[1][1] + self.last_state[1]
            modified_pitch_array = action_type[1][2:] + self.last_state[2]
        #clipping the values now: 
        modified_offset = max(modified_offset, self.last_state[0])
        modified_duration = max(modified_duration, 0)  # Second clipping if it's larger than previous offset value
        modified_pitch_array = np.clip(modified_pitch_array, -1, 127)  # Clip the second action

        # modified pitch array should be rounded
        modified_pitch_array = [round(p) for p in modified_pitch_array]
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)
        
        #check if we reached the end, calculate the reward 
                #0 - 19           #20 
        #get the next value to reward based on the predicted value
        if self.current_step+1 < self.max_generated_steps:
            actual_row = data1.iloc[self.current_step+1]
            self.real_state = (actual_row['offset'], actual_row['duration_ppq'], np.array(ast.literal_eval(actual_row['pitch'])))
            #reward = -calculate_ncd(self.real_state, predicted_state)
            reward = -root_mean_squared_error(np.hstack(self.real_state), np.hstack(predicted_state))
        else:
            #end of dataset
            done = True

        self.current_step +=1
        return self.last_state, predicted_state, reward, done

    def step_simulation(self, action_type):
        print(action_type)
        
        if action_type[0] == 0:
            modified_offset = action_type[1][0] #+ self.last_state[0]
            modified_duration = action_type[1][1] #+ self.last_state[1]
            modified_pitch_array = action_type[1][2:] #+ self.last_state[2]
        else: 
            modified_offset = action_type[1][0] + self.last_state[0]
            modified_duration = action_type[1][1] + self.last_state[1]
            modified_pitch_array = action_type[1][2:] + self.last_state[2]
        
        #clipping the values now:
        modified_offset = max(modified_offset, self.last_state[0])
        modified_duration = max(modified_duration, 0)  # Second clipping if it's larger than previous offset value
        modified_pitch_array = np.clip(modified_pitch_array, -1, 127)  # Clip the second action
        
        # modified pitch array should be rounded
        modified_pitch_array = [round(p) for p in modified_pitch_array]
        predicted_state = (modified_offset, modified_duration, modified_pitch_array)
        
        self.current_step +=1
        return predicted_state
    
    # Access the row corresponding to the current step
    def _get_state(self):
        row = data1.iloc[self.current_step]
        state = (row['offset'], row['duration_ppq'], np.array(ast.literal_eval(row['pitch'])))
        return state


def runAgent(args):
    agent, scoreList = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()

    env = TimeSeriesEnvironment()
    scoreTotal = 0
    episode_length = 20 
    numEpisodes = int(MAX_STEPS_G / episode_length) # 500 / 20 = 25

    direct_steps = round(episode_length * DATA_DIVISION) # 20 * 0.5 = 10
    for ep in range(numEpisodes):
        #memory array is returned as this is the action state
        action_state = env.reset(ep, episode_length) #resets at next 25 window (based on episode)
        #predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        while True:
            #action state is current value, memory based on this will return from get_action_mem (includes it)
            action_state = np.hstack((action_state[:2],action_state[2]))
            
            # change the action_state returning. Either use prediction (recursion) or direct approach 
            
            if ((ep * episode_length) + direct_steps) > env.current_step:
                action_value = (agent.act(action_state))
            else:
                #after direct steps, it will take the predicted value produced then 
                predicted_state = np.hstack((predicted_state[:2],predicted_state[2]))
                action_value = (agent.act(predicted_state))
                
            # Apply clipping to continuous actions : action_value 
            #action_state gives you the next step
            action_state, predicted_state, reward, isDone = env.step(action_value)
            scoreEp += reward
            #env.set_action_mem(action_state) #add state to memory
            if isDone:
                break
        scoreTotal += scoreEp

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal)

    scoreList.append((agent.team.id, agent.team.outcomes))
    return agent, scoreList


def RunBestAgent(args):
    agent, scoreList = args
    # Initialize the environment but starting from the end of the data
    env = TimeSeriesEnvironment(current_step_g=MAX_STEPS_G-1)

    simulation_results = []
    
    print("Priming")
    for x in range(MAX_STEPS_G):
        row = data1.iloc[x]
        state = (row['offset'], row['duration_ppq'], np.array(ast.literal_eval(row['pitch'])))
        action_value = (agent.act(np.hstack((state[:2],state[2]))))[1]
        print(action_value)


    for x in range(EXTRA_TIME_STEPS-1): 

        action_value = (agent.act(np.hstack((env.last_state[:2],env.last_state[2]))))

        action_state = env.step_simulation(action_value)
        print("action_state: ", action_state)
        env.current_step +=1

        new_last_state = (action_state[0], action_state[1], np.array2string(np.array(action_state[2:]), separator=', '))

        simulation_results.append(new_last_state)
        env.last_state = action_state

    new_data = pd.DataFrame(simulation_results, columns=['offset', 'duration_ppq', 'pitch'])
    new_data['pitch'] = new_data['pitch'].apply(str)    
    updated_data = pd.concat([data1[:MAX_STEPS_G], new_data], ignore_index=True)
    updated_data.to_csv('Simulation.csv', index=False)
    
    new_data = pd.DataFrame(simulation_results, columns=['offset', 'duration_ppq', 'pitch'])
    new_data['pitch'] = new_data['pitch'].apply(str)

if __name__ == '__main__':
    tStart = time.time()
    trainer_checkpoint_path = Path("./output_files/trainer_savepoint.pkl")
    gen_checkpoint_path = Path("./output_files/gen_savepoint.txt")

    if trainer_checkpoint_path.exists():
        trainer = pickle.load(open(trainer_checkpoint_path, 'rb'))
        trainer.configFunctions()
        print("LOADED TRAINER")
    else:
        #-159.676
        trainer = Trainer(actions=[MAX_PITCHES+2, MAX_PITCHES+2], teamPopSize=360, pActAtom=1.0, memType="default", operationSet="def")
        gen_start = 0
    
    if gen_checkpoint_path.exists():
        with open(gen_checkpoint_path, 'r') as file:
            gen_start = int(file.read().strip())  # Read the number and convert it to an integer
        print("LOADED GEN NUMBER: ", gen_start)
    else:
        gen_start = 0

    # Open a text file to write output
    with open('./output_files/results.txt', 'a' if gen_start > 0 else 'w') as file:
        file.write(f"Trainer done: {trainer}\n")
        processes = mp.cpu_count()

        man = mp.Manager()
        pool = mp.Pool(processes=processes)
            
        allScores = []

        for gen in range(gen_start, GENERATIONS): 
            scoreList = man.list()
            
            agents = trainer.getAgents()

            pool.map(runAgent, [(agent, scoreList) for agent in agents])
            
            teams = trainer.applyScores(scoreList)  
            
            champ = trainer.getEliteAgent()
            champ.saveToFile("./output_files/best_agent")

            trainer.evolve()
            
            scoreStats = trainer.fitnessStats
            allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
            print(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}")
            file.write(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}\n")

            trainer.saveToFile("./output_files/trainer_savepoint.pkl")
            with open("./output_files/gen_savepoint.txt", 'w') as gen_file:
                gen_file.write(str(gen))
            
            #to keep the champ saved in a file for evaluation later on 
        
        file.write(f'Time Taken (Hours): {(time.time() - tStart)/3600}\n')
        file.write('Final Results:\nMin, Max, Avg\n')
        for score in allScores:
            file.write(f"{score}\n")
        
        #champ = pickle.load(open("./output_files/best_agent", 'rb'))
        champ = trainer.getEliteAgent()
        champ.configFunctionsSelf()

        # Assuming RunBestAgent is a function you have defined earlier
        #empty array is: scorelist
        
        RunBestAgent((champ, []))