import os
import itertools

model_save = '/data/put_data/cmchang/gynecology/model/'					# s
target = 'variability'			# y
length = [300, 500]				# l
# n_channel= 2 					# c
# random_noise = [True, False]	# rn
normalized = [1, 0]		# nm
# l_2 = [1e-6, 1e-4]				# l2
weight_balance = [1, 0] 	# wb
random_state = [13,14,15,16,17] # rs
gpu_id = 0						# g
summary_file = '/data/put_data/cmchang/gynecology/summary.csv' 				# fn

combs = list(itertools.product(length, normalized, weight_balance, random_state))
for para in combs:
	note = os.path.join(target, ('l%s-nm%s-wb%s-rs%s' % (para)))
	model_save_noted = os.path.join(model_save, note)
	script = 'python3 train.py -s {0} -y {1} -g {2} -fn {3}'.format(model_save_noted, target, gpu_id, summary_file)
	script = script + (' -l %s -nm %s -wb %s -rs %s' % (para))
	print(script)