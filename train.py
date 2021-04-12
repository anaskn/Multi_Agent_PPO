from unityagents import UnityEnvironment

import torch
import numpy as np
from collections import deque
import time
import imageio
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO, MemoryBuffer
from env import ContentCaching
import pickle
import matplotlib.pyplot as plt 


def test_data(cpt):

    string1 =  'data/listfile_evol'+str(cpt)+'.data' #_evol'+ , _pos'+
    with open(string1, 'rb') as filehandle:
        # read the data as binary data stream
        lst = pickle.load(filehandle)

    string2 = 'data/nei_tab_pos'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)

    return lst, nei_tab


def train_mappo(lst, nei_tab, cpt, variable, lst_test, nei_tab_test, num_agents=20,steps_per_epoch=20, epochs=100, ttl_var = 3):


    env_name = "Reacher"

    seed =0
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_agents = num_agents
    n_episodes = epochs
    max_steps = steps_per_epoch
    update_interval = 200#16000/n_agents
    log_interval = 10
    solving_threshold = 30
    time_step = 0

    render = False
    train = True
    pretrained = False
    tensorboard_logging = True

    #env = UnityEnvironment(file_name='../Reacher_Windows_x86_64_twenty/Reacher.exe', no_graphics=False)
    #env = UnityEnvironment(file_name="3DBall", seed=1, side_channels=[])
    #env = UnityEnvironment(file_name="./env_binary/env_name", seed=1, side_channels=[config_channel])
    #env = UnityEnvironment(file_name = None)
    env = ContentCaching(num_agents, lst, ttl_var) #env_fn()#, env_fn()

    #brain_name = env.brain_names[0]
    #print("Brain name: ",env.brain_names)
    #brain = env.brains[brain_name]



    #obs_dim = env.observation_space.shape[1]
    #act_dim = env.action_space.shape[0]

    env_info = env.reset()#[brain_name]
    action_size = env.action_space.shape[0]#brain.vector_action_space_size
    states = env.observation_space.shape[1]#env_info.vector_observations
    state_size = states#.shape[1]
    #print("State size: ", state_size)
    #print("Action size: ", action_size)
    #print("state = ", states )

    scores = deque(maxlen=log_interval)
    max_score = -1000
    episode_lengths = deque(maxlen=log_interval)
    rewards =  []

    memory = MemoryBuffer()

    agent = PPO(state_size, action_size)

    if not train:
        agent.policy_old.eval()
    #else:
     #   writer = SummaryWriter(log_dir='logs/'+env_name+'_'+str(time.time()))

    #if pretrained:
     #   agent.policy_old.load_state_dict(torch.load('./'+env_name+'_model_best_old.pth'))
      #  agent.policy.load_state_dict(torch.load('./'+env_name+'_model_best_old.pth'))

    #writerImage = imageio.get_writer('./images/run.gif', mode='I', fps=25)

    def test_agent(nei_tab_test, variable):


        reward_epoch = []
        unused_shared_tab = []
        unused_own_tab = []
        unsatisfied_shared_tab =[]
        unsatisfied_own_tab = []
        num_test_episodes = 1
        for j in range(num_test_episodes):
            reward_step = []
            unused_shared_step = []
            unused_own_step = []
            unsatisfied_shared_step =[]
            unsatisfied_own_step = []
          
            states = env.reset()#[brain_name]
            states = torch.FloatTensor(states)

            for epoch_cpt_test in range(steps_per_epoch):
                # Take deterministic actions at test time (noise_scale=0)
                actions, log_probs = agent.select_action(states)
                
                r, unused_shared, unused_own, unsatisfied_shared, unsatisfied_own = env.step(actions.data.numpy().flatten(),nei_tab, t, variable)
          
                reward_step.append(np.mean(r))
                unused_shared_step.append(unused_shared)
                unused_own_step.append(unused_own)
                unsatisfied_shared_step.append(unsatisfied_shared)
                unsatisfied_own_step.append(unsatisfied_own)

                states = env._next_observation(nei_tab, t, ttl_var)
                states = torch.FloatTensor(states)
              

            reward_epoch.append(np.mean(reward_step))
            unused_shared_tab.append(np.mean(unused_shared_step))
            unused_own_tab.append(np.mean(unused_own_step))
            unsatisfied_shared_tab.append(np.mean(unsatisfied_shared_step))
            unsatisfied_own_tab.append(np.mean(unsatisfied_own_step))


        return reward_epoch, unused_shared_tab , unused_own_tab, unsatisfied_shared_tab, unsatisfied_own_tab



    # test tab
    reward_epoch_tab_test = []
    unused_shared_tab_test = []
    unused_own_tab_test = []
    unsatisfied_shared_tab_test = []
    unsatisfied_own_tab_test = []

    reward_episode = []
    for n_episode in range(1, n_episodes+1):
       
        states = env.reset()#[brain_name]
        states = torch.FloatTensor(states)
        # print("States shape: ", states.shape)
        # state = torch.FloatTensor(state.reshape(1, -1))
        episode_length = 0
        episodic_rewards = []
        reward_step = []
        for t in range(max_steps):
            time_step += 1
            #print(" T  = ", t)

            actions, log_probs = agent.select_action(states)
            

            states = torch.FloatTensor(states)
            memory.states.append(states)
            memory.actions.append(actions)
            memory.logprobs.append(log_probs)

            #actions_tab = []
            # ## Unity env style
            #for agent_id in range(0,20):
            #     actions_tab.append(actions.data.numpy().flatten())



            #print("action = ",actions)
            #print(" sct = ", actions.data.numpy().flatten())
            #print("action = ",len(actions))

            
            rewards, x , y, xx, yy = env.step(actions.data.numpy().flatten(),nei_tab, t, variable)#[brain_name]           # send all actions to tne environment
            reward_step.append(np.mean(rewards))
            states = env._next_observation(nei_tab, t, ttl_var)         # get next state (for each agent)
            #rewards = env_info.rewards                         # get reward (for each agent)
            if t == 19:
                dones = [True]*20#:env_info.local_done
                reward_episode.append(np.mean(reward_step)) 
                reward_epoch_step_test, unused_shared_step_test , unused_own_step_test , unsatisfied_shared_step_test, unsatisfied_own_step_test= test_agent(nei_tab_test, variable)
                reward_epoch_tab_test.append(np.mean(reward_epoch_step_test))
                unused_shared_tab_test.append(np.mean(unused_shared_step_test))
                unused_own_tab_test.append(np.mean(unused_own_step_test))
                unsatisfied_shared_tab_test.append(np.mean(unsatisfied_shared_step_test))
                unsatisfied_own_tab_test.append(np.mean(unsatisfied_own_step_test))

            else:
                dones = [False]*20 
            
            #state = states[0]
            #reward = rewards[0]
            #done = dones[0]

            # state, reward, done, _ = env.step(action.data.numpy().flatten())

            #print(rewards)
            memory.rewards.append(rewards)
            memory.dones.append(dones)
            episodic_rewards.append(rewards)
            state_value = 0
            
            # if render:
            #     image = env.render(mode = 'rgb_array')
                # if time_step % 2 == 0:
                #     writerImage.append_data(image)

            if train:
                if time_step % update_interval == 0:
                    agent.update(memory)
                    time_step = 0
                    memory.clear_buffer()

            episode_length = t
            #print("entryyy")
            if t==19:#
                #print("dones = ", dones)
                break
        
        episode_lengths.append(episode_length)
        total_reward = np.sum(episodic_rewards)/n_agents
        scores.append(total_reward)
        
        if train:
            if n_episode % log_interval == 0:
                print("Episode: ", n_episode, "\t Avg. episode length: ", np.mean(episode_lengths), "\t Avg. score: ", np.mean(scores))

                if np.mean(scores) > solving_threshold:
                    print("Environment solved, saving model")
                    #torch.save(agent.policy_old.state_dict(), 'PPO_model_solved_{}.pth'.format(env_name))
            
            if n_episode % 100 == 0:
                print("Saving model after ", n_episode, " episodes")
                #torch.save(agent.policy_old.state_dict(), '{}_model_{}_episodes.pth'.format(env_name, n_episode))
                
            if total_reward > max_score:
                print("Saving improved model")
                max_score = total_reward
                #torch.save(agent.policy_old.state_dict(), '{}_model_best.pth'.format(env_name))

            #if tensorboard_logging:
             #   writer.add_scalars('Score', {'Score':total_reward, 'Avg._Score': np.mean(scores)}, n_episode)
              #  writer.add_scalars('Episode_length', {'Episode_length':episode_length, 'Avg._Episode length': np.mean(episode_lengths)}, n_episode)
        
        else:
            print("Episode: ", n_episode, "\t Episode length: ", episode_length, "\t Score: ", total_reward)
            
        total_reward = 0

    return reward_epoch_tab_test, unused_shared_tab_test , unused_own_tab_test, reward_episode, unsatisfied_shared_tab_test, unsatisfied_own_tab_test





if __name__ == '__main__':

    cpt = 2
    variable = [8,8,8,8]
    lst_test, nei_tab_test = test_data(9)

    string1 =  'data/listfile_evol'+str(cpt)+'.data' #_evol'+ , _pos'+
    with open(string1, 'rb') as filehandle:
        # read the data as binary data stream
        lst = pickle.load(filehandle)

    string2 = 'data/nei_tab_pos'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)

    reward_test, unused_shared_tab_test , unused_own_tab_test, rewards_train, unsatisfied_shared, unsatisfied_own= train_mappo(lst, nei_tab, cpt, variable, lst_test, nei_tab_test,
                                                                                num_agents=20,steps_per_epoch=20, epochs=1000, ttl_var = 3)



    #print("len = ",len(rewards))
    
    plt.plot(rewards_train, color='blue', marker='v' ,label='Train ') # print reward 
    plt.plot(reward_test, color='red', marker='*' ,label='Test') # print reward 
    
    plt.ylabel('Reward MA-PPO', size= 8 ) #'$U_{nused}$' #Reward
    #plt.xlabel('Episode', size= 10)
    plt.xlabel('Episode ', size= 10)

    plt.xticks(size = 7)
    plt.yticks(size = 7)

    # Add a legend
    plt.legend()

    
    # save file .pdf
    #plt.savefig('plot/0_Reward_plot.pdf') #relusigmoid


    plt.show()

    plt.close()
    print("End")

        