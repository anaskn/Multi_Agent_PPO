import matplotlib.pyplot as plt
import time
import os
import signal
from subprocess import Popen, PIPE
import json
import gym
import json
import datetime as dt
from gym import spaces
import numpy as np
import pandas as pd
import random
import statistics






class ContentCaching(gym.Env):	

	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	#env init
	def __init__(self,num_agent, lst, ttl_var):

		super(ContentCaching, self).__init__()

		#Reward
		self.reward_range = (-1000, 1000) 

		#Action_space
		self.action_space = spaces.Box(low= 0, high= 1 ,shape=(1,), dtype=np.float32)

		#Observation_space:
		self.observation_space = spaces.Box(low= 0, high= 100, shape=(1,3), dtype=np.float32)

		# Reset the state of the environment to an initial state
		
		tab_cache= []
		tab_request = []
		nei_req = []
		cache_on_tab = []
		neighbor_number_tab = []
		ttl_tab = []
		for xx in range(num_agent):
			tab_cache.append(50) #lst[1][xx] 
			tab_request.append(lst[xx]) # lst[0][xx]
			nei_req.append(-99)
			cache_on_tab.append(0)
			neighbor_number_tab.append(0)
			ttl_tab.append(np.zeros(20))#ttl_var



		
		self.caching_cap =  tab_cache 
		self.request = tab_request 
		self.neigbors_request = nei_req 
		self.cache_on = cache_on_tab
		self.neighbor_number = neighbor_number_tab
		self.ttl = ttl_tab

	

	def reset(self):
		entity_pos = []
		for x in range(len(self.caching_cap)):
			lstt= []
			lstt.append(self.caching_cap[x])
			lstt.append(self.request[0][x])
			lstt.append(self.neigbors_request[x])
			entity_pos.append(lstt)

	
		entity_pos = np.array(entity_pos)
		return entity_pos

	def _next_observation(self, nei_tab, i, ttl_var):

		
		entity_pos = []
		unsatisfied_lst = []
		
		for x in range(len(self.caching_cap)):

			lstt = []

			lstt.append(self.request[i][x])

			#init  caching_cap
			#"""
			if i == 0 :
				#self.ttl[x]=np.zeros(20)
				lstt.append(50)
				self.caching_cap[x]=50

			else:			
				if i-ttl_var > 0:
					#print("i == ", i)
					self.caching_cap[x] = self.caching_cap[x] + self.ttl[x][i-ttl_var]
				#if x== 3 :
				#	print("self.ttl[x][i-ttl_var] == ", self.ttl[x])
				#	print("self.caching_cap[x] = ", self.caching_cap[x], " i = ", i)


				min_val = min(self.caching_cap[x] , float(self.cache_on[x]))
				self.caching_cap[x] = self.caching_cap[x] - min_val	 
				self.ttl[x][i] = min_val
				lstt.append(self.caching_cap[x])

			"""
			#init  caching_cap
			if i == 0 :
				lstt.append(50) #self.caching_cap[x]
				#self.ttl[x]=np.zeros(20)
			else:
				
				self.caching_cap[x] = self.caching_cap[x] + self.ttl[x][i%ttl_var]
				if x== 3 :
					print("self.ttl[x][i-ttl_var] == ", self.ttl[x])

				#unsatisfied
				#if self.caching_cap[x] < float(self.cache_on[x]) :
				#	unsatisfied_lst.append(float(self.cache_on[x])-self.caching_cap[x])
					
				#else:
				#	unsatisfied_lst.append(0)

				min_val = min(self.caching_cap[x] , float(self.cache_on[x]))
				self.caching_cap[x] = self.caching_cap[x] - min_val
				self.ttl[x][i%ttl_var] = min_val
				lstt.append(self.caching_cap[x])
			"""



			
			#init  neigbors_request
			cache = 0
			for y in range(len(nei_tab[i][x])): # neighbor of neighbor len

				if len(nei_tab[i][y]) == 0:
					cache = cache + 0
				
				else:
					cache = cache + (self.request[i][nei_tab[i][x][y]]/len(nei_tab[i][nei_tab[i][x][y]]) )


 
			if len(nei_tab[i][x])==0: # test if neighbor len == 0 
				self.neigbors_request[x]= 0
				self.neighbor_number[x] = 0
				lstt.append(0)
			else:
				self.neigbors_request[x] = cache/len(nei_tab[i][x])
				self.neighbor_number[x] = len(nei_tab[i][x])
				lstt.append(self.neigbors_request[x])  #cache/len(nei_tab[i][x])

			#print("lsssttt = ", lstt)
			entity_pos.append(lstt)

		entity_pos = np.array(entity_pos)

		return entity_pos#, np.mean(unsatisfied_lst)



	def step(self, action, nei_tab, i, variable):
		#print("i = ", i)
		reward=[]
		R_c = variable[0]
		C_o = variable[1]
		C_u = variable[2]
		fact_k = variable[3]	

		unused_shared = []
		unused_own = []
		nei_request_tab = []

		# compute : sum(D_j(1-a_j)*G_j)/N_j
		for zz in range(len(action)):

			#print("action before = ",action[zz])		

			cache1 = 0
			for y in range(len(nei_tab[i][zz])):

				if len(nei_tab[i][y]) == 0:
					cache1= cache1 + 0
				else :
					try:
						cache1=cache1+(max(0,(self.request[i][nei_tab[i][zz][y]]-((1-action[nei_tab[i][zz][y]])*self.caching_cap[nei_tab[i][zz][y]]))/len(nei_tab[i][nei_tab[i][zz][y]])) )

					except:

						action = action[0]
						cache1=cache1+(max(0,(self.request[i][nei_tab[i][zz][y]]-((1-action[nei_tab[i][zz][y]])*self.caching_cap[nei_tab[i][zz][y]]))/len(nei_tab[i][nei_tab[i][zz][y]])) )
						



						


			if len(nei_tab[i][zz]) == 0 :
				cache1 = 0
			#else:
			#	cache1 = cache1/len(nei_tab[i][zz])

			nei_request_tab.append(cache1)



			#reward computing
			#"""
			try:
				f = R_c * max(0, (1-action[zz]) * self.caching_cap[zz] )  \
				   - C_u * ( max(0,  (self.request[i][zz]-(action[zz]*self.caching_cap[zz]))) + max(0, ( cache1 - (1-action[zz])*self.caching_cap[zz])/fact_k)  ) \
				      - C_o * ( max(0, ((action[zz]*self.caching_cap[zz])-self.request[i][zz])/fact_k) + max (0, ((1-action[zz])*self.caching_cap[zz]) - cache1) )		
			except:
				action = action[0]
				f = R_c * max(0, (1-action[zz]) * self.caching_cap[zz] )  \
				   - C_u * ( max(0,  (self.request[i][zz]-(action[zz]*self.caching_cap[zz]))) + max(0, ( cache1 - (1-action[zz])*self.caching_cap[zz])/fact_k)  ) \
				      - C_o * ( max(0, ((action[zz]*self.caching_cap[zz])-self.request[i][zz])/fact_k) + max (0, ((1-action[zz])*self.caching_cap[zz]) - cache1) )	
			"""
			#print("f == ", float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1 )) )
			#print("f2 = ", float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] )))
			#f = float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1 )) + float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] ))
			try:
				f =  float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1  )) + float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] ))
			except:
				action = action[0]
				f =  float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1  )) + float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] ))
			"""
			
			
			#unused_shared
			unused_shared.append( float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1 )) + float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] )) )
			#unused_shared.append( float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1  )))
			

			#unused_own
			unused_own.append( float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] )))
			#print("f2 = ", float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] )))
			#print("unused_own = ", float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] )))





			#try:
			reward.append(f)#[0])
			#except:
			#reward.append(f)

		nei_demand = []
		my_demand  = []
		nei_number = []

		
		#provisoires computing my_demand / neighbor demand/ neighbor number
		for zz in range(len(action)):

			#my demand 
			my_demand.append(self.request[i][zz])

			# neighbor demand sum 
			nei_demand.append(self.neigbors_request[zz])

			# my neighbor number sum
			nei_number.append(self.neighbor_number[zz])


		#init  self.cache_on[x] : compute the amount of resources from own demand and neighbot to cache 
		for zz in range(len(action)):
			self.cache_on[zz] = min(self.request[i][zz], ((action[zz]*100) * self.caching_cap[zz]) / 100.0)  \
				+ min(self.neigbors_request[zz], (((1-action[zz])*100) * self.caching_cap[zz]) / 100.0)
			

		return 	reward, statistics.mean(unused_shared), statistics.mean(unused_own)#, np.mean(nei_demand), np.mean(my_demand), np.mean(nei_number)

	 















	
"""
if __name__ == '__main__':
	print("topo begin")
	net = topology()
	to_obs = obs_init()
	data = np.array([net.caching_cap, net.computing_cap, net.bandwith_cap ])
	to_pred = pd.DataFrame({'caching': data[0], 'computing': data[1], 'bandwith': data[2]})
	env = ContentCaching(to_obs , to_pred, net)
	obs = env._next_observation(net)
	obs = env.reset(net)
	
			
				for i in range(1):
			
					action, _states = env.model.predict(obs)
					obs, rewards, done, info = env.step(action)
					env.render()
			
				print(obs)
			
	
	stopping(net)
"""