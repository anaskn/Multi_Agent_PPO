from train import train_mappo


import sys


import pickle
import matplotlib.pyplot as plt 
import statistics

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


if __name__ == '__main__':



	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--steps', type=int, default=20)
	parser.add_argument('--epochs', type=int, default=200)#400
	parser.add_argument('--num_test_episodes', type=int, default=1)
	parser.add_argument('--ttl_var', type=int, default=5)
	args = parser.parse_args()


	lst_test, nei_tab_test = test_data(9)

	#time.sleep(18000)
	#Transfer Learning 


	y=0
	parameters = [[y, 8, 8, 4] , [8, y, 8, 4], [8, 8, y, 4], [8, 8, 8, y]]

	pdf_plot = ["R_c", "C_o", "C_u", "k"]

	lst=0
	want = [0,1,2,3]

	algo_lst = ["mappo"]



	for algo in algo_lst:
	#for para in want:#want 

		#print(para)
		algo_reward_test = []
		algo_unused_shared = []
		algo_unused_own = []
		algo_reward_test_cut = []
		algo_unused_shared_cut = []
		algo_unused_own_cut = []

		variable = [1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60] #[1,10,20,60,150,400,700,1000] #
		#variable = [1,100]


		

		#for algo in algo_lst:
		for para in range(len(pdf_plot)):#want:#want 

			scoress = []

			total_reward_test= []
			total_unused_shared = []
			total_unused_own = []

			total_reward_test_cut = []
			total_unused_shared_cut = []
			total_unused_own_cut = []


			
				
			for x in range(len(variable)):
				loop_reward_test = []
				loop_unused_shared = []
				loop_unused_own = []
				loop_reward_test_cut = []
				loop_unused_shared_cut = []
				loop_unused_own_cut = []


				for loop in range(2):#loop over each variable ##########

					parameters[para][para]= variable[x]
					print("parameters[para] = ",  parameters[para])

					all_reward_test = []
					all_unused_shared = []
					all_unused_own = []

					for cpt in range(1,5): #1,10 1,2

						print("calcul of : "+pdf_plot[para], " scenario num : ", cpt, " for the value : ", variable[x]  )

						string1 =  'data/listfile_evol'+str(cpt)+'.data' #_evol'+ , _pos'+
						with open(string1, 'rb') as filehandle:
							# read the data as binary data stream
							lst = pickle.load(filehandle)

						string2 = 'data/nei_tab_pos'+str(cpt)+'.data'
						with open(string2, 'rb') as filehandle:
							# read the data as binary data stream
							nei_tab = pickle.load(filehandle)

						if algo == 'ppo':
						#ppo
							import spinup.algos.pytorch.ppo.core as core
							reward_test, unused_shared, unused_own, reward_train = ppo(lambda : gym.make(args.env), lst, nei_tab, cpt, parameters[para], lst_test, nei_tab_test,
																	actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
																	gamma=args.gamma, seed=args.seed, epochs=args.epochs, ttl_var = args.ttl_var, 
																	num_test_episodes= args.num_test_episodes, logger_kwargs=logger_kwargs)
						if algo == 'mappo':
							#ddpg
							reward_test, unused_shared, unused_own, reward_train = train_mappo(lst, nei_tab, cpt, parameters[para], lst_test, nei_tab_test, num_agents=20,
																	steps_per_epoch=args.steps, epochs=args.epochs, ttl_var = args.ttl_var)
							
    

						all_reward_test.append(reward_test)
						all_unused_shared.append(unused_shared)
						all_unused_own.append(unused_own)


					loop_reward_test_cut.append(statistics.mean(all_reward_test[3]))
					loop_unused_shared_cut.append(statistics.mean(all_unused_shared[3]))
					loop_unused_own_cut.append(statistics.mean(all_unused_own[3]))


					all_reward_test = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_reward_test[0], all_reward_test[1], all_reward_test[2], all_reward_test[3])]
					all_unused_shared = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unused_shared[0], all_unused_shared[1], all_unused_shared[2], all_unused_shared[3])]
					all_unused_own = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unused_own[0], all_unused_own[1], all_unused_own[2], all_unused_own[3])]

					loop_reward_test.append(statistics.mean(all_reward_test))
					loop_unused_shared.append(statistics.mean(all_unused_shared))
					loop_unused_own.append(statistics.mean(all_unused_own))


				total_reward_test.append(statistics.mean(loop_reward_test))
				total_unused_shared.append(statistics.mean(loop_unused_shared))
				total_unused_own.append(statistics.mean(loop_unused_own))


				total_reward_test_cut.append(statistics.mean(loop_reward_test_cut))
				total_unused_shared_cut.append(statistics.mean(loop_unused_shared_cut))
				total_unused_own_cut.append(statistics.mean(loop_unused_own_cut))

			algo_reward_test.append(total_reward_test)
			algo_unused_shared.append(total_unused_shared)
			algo_unused_own.append(total_unused_own)


			algo_reward_test_cut.append(total_reward_test_cut)
			algo_unused_shared_cut.append(total_unused_shared_cut)
			algo_unused_own_cut.append(total_unused_own_cut)
				

		
		times = [1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60]
		#times = [1,100]
		#times = [2,4,6,8,10,12,14,16,18,20]
		#print("algo_reward_test == ", algo_reward_test)
		#print("total_unused_shared == ", algo_unused_shared)
			

		#plt.plot(reward, color='blue', marker='v' ,label='DDPG Spinning Reward') # print reward
		
		plt.plot(times , algo_unused_shared[0], color='orange', linestyle='dotted', marker='x' ,label='$R_c$') #  unused shared  
		plt.plot(times , algo_unused_shared[1], color='purple', linestyle='-', marker='+' ,label='$C_o$') # unused own 
		plt.plot(times , algo_unused_shared[2], color='red', linestyle='dashed', marker='D' ,label='$C_u$') #  unused shared  
		plt.plot(times , algo_unused_shared[3], color='green', linestyle='dashdot', marker='*' ,label='$k$') # unused own 
		
		plt.ylabel('', size= 8 ) #'$U_{nused}$' #Reward
		#plt.xlabel('Episode', size= 10)
		plt.xlabel('$'+pdf_plot[para]+'$', size= 10)

		plt.xticks(size = 7)
		plt.yticks(size = 7)

		# Add a legend
		plt.legend()

		
		# save file .pdf
		plt.savefig('plot/z1_five_all_mappo_plot.pdf') #relusigmoid


		#to stock data 
		our_file = [algo_unused_shared[0],algo_unused_shared[1],algo_unused_shared[2],algo_unused_shared[3]]#, algo_unused_shared[1], algo_unused_own[1], algo_unused_shared[2], algo_unused_own[2] ]
		with open('model/z1_five_all_mappo.data', 'wb') as filehandle: #ddpg4442 #ddpg6664
		#  # store the data as binary data stream
			pickle.dump(our_file, filehandle)
		
		#print("algo_unused_shared = ", len(algo_unused_shared))
		
		#plt.show()

		plt.close()
		print("End")

		
		"""
		#plot only the last one 
		plt.plot(times , algo_unused_shared_cut[0], color='orange', linestyle='dotted', marker='x' ,label='$R_c$') #  unused shared  
		#plt.plot(times , algo_unused_shared_cut[1], color='purple', linestyle='-', marker='+' ,label='$C_o$') # unused own 
		#plt.plot(times , algo_unused_shared_cut[2], color='red', linestyle='dashed', marker='D' ,label='$C_u$') #  unused shared  
		#plt.plot(times , algo_unused_shared_cut[3], color='green', linestyle='dashdot', marker='*' ,label='$k$') # unused own 
		

		plt.ylabel('', size= 8 ) #'$U_{nused}$' #Reward
		plt.xlabel('$'+pdf_plot[para]+'$', size= 10)

		plt.xticks(size = 7)
		plt.yticks(size = 7)

		# Add a legend
		plt.legend()
		
		# save file .pdf
		#plt.savefig('plot/07_last_all_'+pdf_plot[para]+'_plot.pdf') #relusigmoid


		#to stock data 
		#our_file = [algo_unused_shared_cut[0], algo_unused_shared_cut[1],algo_unused_shared_cut[2],algo_unused_shared_cut[3]]#, algo_unused_shared_cut[1], algo_unused_own_cut[1], algo_unused_shared_cut[2], algo_unused_own_cut[2]]
		#with open('model/07_last_rc_all_'+pdf_plot[para]+'.data', 'wb') as filehandle: 
		#  # store the data as binary data stream
		#	pickle.dump(our_file, filehandle)
		
		

		plt.show()

		plt.close()
		print("End")

		"""
		
		