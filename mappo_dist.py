from train import train_mappo
import sys
import time
import pickle
import matplotlib.pyplot as plt 
import statistics

def test_data(cpt,dist_para):

    string1 =  'data3/listfile_evol_'+str(cpt)+'_'+str(dist_para)+'.data' #_evol'+ , _pos'+
    with open(string1, 'rb') as filehandle:
        # read the data as binary data stream
        lst = pickle.load(filehandle)

    string2 = 'data3/nei_tab_pos'+str(cpt)+'_'+str(dist_para)+'.data'
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
	parameters = [[y, 8, 8, 4]]# , [8, y, 8, 4], [8, 8, y, 4], [8, 8, 8, y]]

	pdf_plot = ["R_c"]#, "C_o", "C_u", "k"]

	lst=0
	want = [0,1,2,3]

	for para in range(len(parameters)):#want 

		print(para)

		#variable = [1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60] #[1,10,20,60,150,400,700,1000] #
		variable = [8]



		scoress = []

		algo_reward_test = []
		algo_unused_shared = []
		algo_unused_own = []
		algo_unsatisfied_shared =[]
		algo_unsatisfied_own =[]


		algo_reward_test_cut = []
		algo_unused_shared_cut = []
		algo_unused_own_cut = []


		algo_lst = ["mappo"]

		for algo in algo_lst:


			total_reward_test= []
			total_unused_shared = []
			total_unused_own = []
			total_unsatisfied_shared =[]
			total_unsatisfied_own =[]

			total_reward_test_cut = []
			total_unused_shared_cut = []
			total_unused_own_cut = []






			dist_nei = [10,20,30,40,50,60]

			for dist_para in dist_nei :

				lst_test, nei_tab_test = test_data(5,dist_para)
					
				for x in range(len(variable)):

					loop_reward_test = []
					loop_unused_shared = []
					loop_unused_own = []
					loop_unsatisfied_shared = []
					loop_unsatisfied_own = []
					loop_reward_test_cut = []
					loop_unused_shared_cut = []
					loop_unused_own_cut = []


					for loop in range(2):#loop over each variable##########################

						parameters[para][para]= variable[x]

						all_reward_test = []
						all_unused_shared = []
						all_unused_own = []
						all_unsatisfied_shared=[]
						all_unsatisfied_own =[]

						for cpt in range(1,5): #1,10 1,2

							#print("calcul of : "+pdf_plot[para], " scenario num : ", cpt, " for the value : ", variable[x]  )

							string1 =  'data3/listfile_evol_'+str(cpt)+'_'+str(dist_para)+'.data' #_evol'+ , _pos'+
							with open(string1, 'rb') as filehandle:
								# read the data as binary data stream
								lst = pickle.load(filehandle)

							string2 = 'data3/nei_tab_pos'+str(cpt)+'_'+str(dist_para)+'.data'
							with open(string2, 'rb') as filehandle:
								# read the data as binary data stream
								nei_tab = pickle.load(filehandle)

							if algo == 'mappo':
								#ddpg
								reward_test, unused_shared, unused_own, reward_train, unsatisfied_shared, unsatisfied_own = train_mappo(lst, nei_tab, cpt, parameters[para], lst_test, nei_tab_test, num_agents=20,
																		steps_per_epoch=args.steps, epochs=args.epochs, ttl_var = args.ttl_var)



							all_reward_test.append(reward_test)
							all_unused_shared.append(unused_shared)
							all_unused_own.append(unused_own)
							all_unsatisfied_shared.append(unsatisfied_shared)
							all_unsatisfied_own.append(unsatisfied_own)



						loop_reward_test_cut.append(statistics.mean(all_reward_test[3]))
						loop_unused_shared_cut.append(statistics.mean(all_unused_shared[3]))
						loop_unused_own_cut.append(statistics.mean(all_unused_own[3]))

						all_reward_test = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_reward_test[0], all_reward_test[1], all_reward_test[2], all_reward_test[3])]
						all_unused_shared = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unused_shared[0], all_unused_shared[1], all_unused_shared[2], all_unused_shared[3])]
						all_unused_own = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unused_own[0], all_unused_own[1], all_unused_own[2], all_unused_own[3])]
						all_unsatisfied_shared = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unsatisfied_shared[0], all_unsatisfied_shared[1], all_unsatisfied_shared[2], all_unsatisfied_shared[3])]
						all_unsatisfied_own = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unsatisfied_own[0], all_unsatisfied_own[1], all_unsatisfied_own[2], all_unsatisfied_own[3])]
					
		
						loop_reward_test.append(statistics.mean(all_reward_test))
						loop_unused_shared.append(statistics.mean(all_unused_shared))
						loop_unused_own.append(statistics.mean(all_unused_own))
						loop_unsatisfied_shared.append(statistics.mean(all_unsatisfied_shared))
						loop_unsatisfied_own.append(statistics.mean(all_unsatisfied_own))


					total_reward_test.append(statistics.mean(loop_reward_test))
					total_unused_shared.append(statistics.mean(loop_unused_shared))
					total_unused_own.append(statistics.mean(loop_unused_own))
					total_unsatisfied_shared.append(statistics.mean(loop_unsatisfied_shared))
					total_unsatisfied_own.append(statistics.mean(loop_unsatisfied_own))


					total_reward_test_cut.append(statistics.mean(loop_reward_test_cut))
					total_unused_shared_cut.append(statistics.mean(loop_unused_shared_cut))
					total_unused_own_cut.append(statistics.mean(loop_unused_own_cut))



					

			algo_reward_test.append(total_reward_test)
			algo_unused_shared.append(total_unused_shared)
			algo_unused_own.append(total_unused_own)
			algo_unsatisfied_shared.append(total_unsatisfied_shared)
			algo_unsatisfied_own.append(total_unsatisfied_own)

			algo_reward_test_cut.append(total_reward_test_cut)
			algo_unused_shared_cut.append(total_unused_shared_cut)
			algo_unused_own_cut.append(total_unused_own_cut)

				


		#times = [1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60]
		times = [10,20,30,40,50,60]	
		

		#plt.plot(reward, color='blue', marker='v' ,label='DDPG Spinning Reward') # print reward


		plt.plot(times , algo_unused_shared[0], color='orange', linestyle='dotted', marker='x' ,label='mappo_$Unused_{g}$') #  unused shared  'ppo_$Unused$'
		plt.plot(times , algo_unused_own[0], color='purple', linestyle='-', marker='+' ,label='mappo_$Unused_{o}$') # unused own 
		#plt.plot(times , algo_unused_shared[1], color='red', linestyle='dashed', marker='D' ,label='trpo_$Unused_{g}$') #  unused shared  
		#plt.plot(times , algo_unused_own[1], color='green', linestyle='dashdot', marker='*' ,label='trpo_$Unused_{o}$') # unused own 


		plt.ylabel('Unused caching resources', size= 8 ) #'$U_{nused}$' #Reward
		#plt.xlabel('Episode', size= 10)
		plt.xlabel('Communication Range', size= 10)

		plt.xticks(size = 7)
		plt.yticks(size = 7)

		# Add a legend
		plt.legend()
	
		
		# save file .pdf

		plt.savefig('plot/99_unused_Dist_mappo.pdf') #relusigmoid


		#to stock data 
		our_file = [algo_unused_shared[0],algo_unused_own[0]]#,algo_unused_shared[1],algo_unused_own[1]]#, algo_unused_shared[1], algo_unused_own[1], algo_unused_shared[2], algo_unused_own[2] ]
		with open('model/99_unused_Dist_mappo.data', 'wb') as filehandle: #07_five_rc_all_'+pdf_plot[para]+'
		#  # store the data as binary data stream
			pickle.dump(our_file, filehandle)
		
		
		#plt.show()

		plt.close()
		print("End")

		

		#plot only the last one 
		plt.plot(times , algo_unsatisfied_shared[0], color='orange', linestyle='dotted', marker='x' ,label='mappo_$Unsatisfied_{g}$') #  unused shared  
		plt.plot(times , algo_unsatisfied_own[0], color='purple', linestyle='-', marker='+' ,label='mappo_$Unsatisfied_{o}$') # unused own 
		#plt.plot(times , algo_unsatisfied_shared[1], color='red', linestyle='dashed', marker='D' ,label='trpo_$Unsatisfied_{g}$') #  unused shared  
		#plt.plot(times , algo_unsatisfied_own[1], color='green', linestyle='dashdot', marker='*' ,label='trpo_$Unsatisfied_{o}$') # unused own 


		plt.ylabel('Unsatisfied caching demands', size= 8 ) #'$U_{nused}$' #Reward
		plt.xlabel('Communication Range', size= 10)

		plt.xticks(size = 7)
		plt.yticks(size = 7)

		# Add a legend
		plt.legend()
		
		# save file .pdf
		plt.savefig('plot/99_unsatisfied_Dist_mappo.pdf') #relusigmoid


		#to stock data 
		our_file = [algo_unsatisfied_shared[0], algo_unsatisfied_own[0]]#,algo_unsatisfied_shared[1],algo_unsatisfied_own[1]]#, algo_unused_shared_cut[1], algo_unused_own_cut[1], algo_unused_shared_cut[2], algo_unused_own_cut[2]]
		with open('model/99_unsatisfied_Dist_mappo.data', 'wb') as filehandle: 
		  # store the data as binary data stream
			pickle.dump(our_file, filehandle)
		
		

		#plt.show()

		plt.close()
		print("End")

		




