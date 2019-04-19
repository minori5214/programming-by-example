import sys
sys.path.append("C:\\Users\\minori\\Desktop\\PBE_codes")

import itertools
import numpy as np
import copy
import tools_pbe as tls
from statistics import mode

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def data_category_to_num(x_train, key, x_test=None):
	#Get unique values
	values_tr = list(x_train[key].unique())
	if x_test is not None:
		values_te = list(x_test[key].unique())
		values = list(set(values_tr + values_te))
	else:
		values = values_tr

	#Make dictionary
	title_mapping = dict()
	for i, value in enumerate(values):
		title_mapping[value] = i
	print(title_mapping)

	#Convert into number
	if x_test is not None:
		combine = [x_train, x_test]
	else:
		combine = [x_train]
	for dataset in combine:
		dataset[key] = dataset[key].map(title_mapping)

	return x_train, x_test, combine

def conjunction_cost(constructor, costs_dict):
	cost = 0
	for match in constructor.Matches:
		for token in match.TokenSeq:
			cost += costs_dict[token]

	return cost

def substr_cost(constructor, costs_dict):
	cost = costs_dict[constructor.Token]

	return cost

def argument_cost(constructor):
	costs_dict = {'StartTok': 1, 'EndTok': 1, 'AlphaTok': 2, 'NumTok': 2, 'SpaceTok':3, 'PeriodTok':3, 'CommaTok':3, \
				'LeftParenthesisTok':3, 'RightParenthesisTok':3, 'SQuoteTok':3, 'DQuoteTok':3, 'HyphenTok':3, \
				'UBarTok':3, 'SlashTok':3, 'NoneTok':3}
	if constructor.id == "Conjunction":
		cost = conjunction_cost(constructor, costs_dict)
	elif constructor.id == "SubStr":
		cost = substr_cost(constructor, costs_dict)
	elif constructor.id == "ConstStr":
		cost = 3

	return cost

def token_num_count(constructor, Token):
	num = 0
	if constructor.id == "Conjunction":
		for match in constructor.Matches:
			for token in match.TokenSeq:
				if token == Token:
					num += 1
	elif constructor.id == "SubStr":
		if constructor.Token == Token:
			num += 1
	elif constructor.id == "ConstStr":
		num = 0

	return num

def recursive_sort(token_idx, trace_exprs, switches):
	token_list = ['StartTok', 'EndTok', 'AlphaTok', 'NumTok', 'SpaceTok', 'PeriodTok', 'CommaTok', \
				'LeftParenthesisTok', 'RightParenthesisTok', 'SQuoteTok', 'DQuoteTok', 'HyphenTok', \
				'UBarTok', 'SlashTok', 'NoneTok']

	trace_exprs_sort, switches_sort = list(), list() #[[trace],[trace],[trace],[trace]], [[swi],[swi],[swi],[swi]]

	token_nums = list()
	for switch, trace_expr in zip(switches, trace_exprs):
		token_num_cond, token_num_trace = 0, 0
		for cond, trace in zip(switch, trace_expr):
			token_num_cond += token_num_count(cond, token_list[token_idx])
			token_num_trace += token_num_count(trace, token_list[token_idx])
		token_num = [token_num_cond, token_num_trace]
		token_nums.append(token_num)
	token_nums = np.array(token_nums)
	token_nums_sum = token_nums.sum(axis=1) #[1,4,4,2]

	nums = np.unique(token_nums_sum)
	nums_sort = nums[np.argsort(nums)[::-1]] #[4,2,1]
	for num in nums_sort:
		part_trace_expr, part_switch = list(), list()
		for i in range(0, len(token_nums_sum)):
			if token_nums_sum[i] == num:
				part_trace_expr.append(trace_exprs[i])
				part_switch.append(switches[i])

		assert len(part_trace_expr) == len(part_switch) > 0, 'Assersion Error: Input[{0}, {1}]'.format(len(part_trace_expr), len(part_switch))
		if len(part_trace_expr) != 1:
			part_trace_expr, part_switch = recursive_sort(token_idx+1, part_trace_expr, part_switch)
			trace_exprs_sort.extend(part_trace_expr)
			switches_sort.extend(part_switch)			
		elif len(part_trace_expr) == 1:
			trace_exprs_sort.extend(part_trace_expr)
			switches_sort.extend(part_switch)

	return trace_exprs_sort, switches_sort #[[trace],[trace],[trace]], [[swi],[swi],[swi]]

def count_argnum(conjunction):
	num = 0
	for match in conjunction.Matches:
		num += len(match.TokenSeq) * match.num
	return num

def RANKING_FLASHFILL(trace_exprs_val, switches_val):
	# Prefer fewer arguments in conditionals and trace expressions
	arg_nums = list()
	for switch, trace_expr in zip(switches_val, trace_exprs_val):
		arg_num = 0
		for cond, trace in zip(switch, trace_expr):
			# Prefer fewer numbers of Sum of Token in switch + trace_expr
			arg_num += count_argnum(cond) #sum([len(match.TokenSeq) for match in cond.Matches])
			arg_num += 1 #len(trace.args)
		arg_nums.append(arg_num)
	arg_nums = np.array(arg_nums)
	arg_min_idxs = np.where(arg_nums == arg_nums.min())[0]

	switches_val2, trace_exprs_val2 = list(), list()
	for i in arg_min_idxs:
		switches_val2.append(switches_val[i])
		trace_exprs_val2.append(trace_exprs_val[i])

	#Prefer rule that has more SubStr constructor
	substr_nums = list()
	for trace_expr in trace_exprs_val2:
		substr_num = 0
		for trace in trace_expr:
			if trace.id == "SubStr":
				substr_num += 1
		substr_nums.append(substr_num)
	substr_nums = np.array(substr_nums)
	substr_max_idxs = np.where(substr_nums == substr_nums.max())[0]

	switches_val3, trace_exprs_val3 = list(), list()
	for i in substr_max_idxs:
		switches_val3.append(switches_val2[i])
		trace_exprs_val3.append(trace_exprs_val2[i])

	#Prefer rule that has less match constructor
	match_nums = list()
	for switch in switches_val3:
		match_num = 0
		for cond in switch:
			match_num += len(cond.Matches)
		match_nums.append(match_num)
	match_nums = np.array(match_nums)
	match_min_idxs = np.where(match_nums == match_nums.min())[0]

	switches_val4, trace_exprs_val4 = list(), list()
	for i in match_min_idxs:
		switches_val4.append(switches_val3[i])
		trace_exprs_val4.append(trace_exprs_val3[i])

	#Prefer larger character class arguments
	arg_costs = list()
	for switch, trace_expr in zip(switches_val4, trace_exprs_val4):
		arg_cost = 0
		for cond, trace in zip(switch, trace_expr):
			arg_cost += argument_cost(cond)
			arg_cost += argument_cost(trace)
		arg_costs.append(arg_cost)
	arg_costs = np.array(arg_costs)
	cost_min_idxs = np.where(arg_costs == arg_costs.min())[0]

	switches_val5, trace_exprs_val5 = list(), list()
	for i in cost_min_idxs:
		switches_val5.append(switches_val4[i])
		trace_exprs_val5.append(trace_exprs_val4[i])

	#Print all domain-specific languages
	print(len(trace_exprs_val5))
	for trace_exprs, switches in zip(trace_exprs_val5, switches_val5):
		for trace_expr, switch in zip(trace_exprs, switches):
			trace_expr.print_constructor()
			switch.print_constructor()
			print("")
		print("")

	#Final match
	trace_exprs_sort, switches_sort = recursive_sort(0, trace_exprs_val5, switches_val5)

	#Print all domain-specific languages
	i = 0
	print("(FLASHFILL) ---ADOPTED DOMAIN-SPECIFIC LANGUAGES---")
	for trace_exprs, switches in zip(trace_exprs_sort, switches_sort):
		print("Domain-specific language {}: ".format(i))
		for trace_expr, switch in zip(trace_exprs, switches):
			print("Conditional (IF):")
			switch.print_constructor()
			print("Trace expression (THEN): ")
			trace_expr.print_constructor()
			print("")
		i+=1
		print("")

	return trace_exprs_sort[0], switches_sort[0]

def RANKING(trace_exprs_val, switches_val, data, target, target_name, seed=0, test_size=0.0):
	print(data.describe(include=['O']))
	accs = list()

	#Data transformation for each rule
	for switch, trace_expr in zip(switches_val, trace_exprs_val):
		new_dat = list(copy.deepcopy(data))
		counter = 0
		exceptions = list()
		valuelist = list()
		count = 0
		for i in range(0, len(new_dat)):
			for cond, trace in zip(switch, trace_expr):
				if cond.Conditional(new_dat[i]):
					expr = copy.deepcopy(trace)
					expr.String = new_dat[i]
					new_dat[i] = expr.get_value()
					valuelist.append(new_dat[i])
					count += 1
			if count == 0:
				exceptions.append(i)
			count = 0

		#Show edge case
		#if len(exceptions) != 0:
		#	print("Value(s) not handled:")
		#	print(list(set([new_dat[idx] for idx in exceptions])))
		#print("Alternative value: ", mode(valuelist))

		#Replace exceptions (that cannot be handled by synthesized program) by mode
		repl = mode(valuelist)
		for idx in exceptions:
			new_dat[idx] = repl

		new_dat = np.array(new_dat)

		X = pd.DataFrame(data=new_dat, columns=['new_dat'], dtype=type(new_dat[0]))
		X, _, _ = data_category_to_num(X, 'new_dat')

		x_train, x_test, y_train, y_test = train_test_split(X, target, test_size=test_size, shuffle=False)

		random_forest = RandomForestClassifier(n_estimators=150, random_state=seed)
		random_forest.fit(x_train, y_train[target_name])
		acc_random_forest = round(random_forest.score(x_train, y_train[target_name]) * 100, 2)
		print(acc_random_forest)
		accs.append(acc_random_forest)
		counter += 1
	idx = np.argmax(accs)

	#Print selected domain-specific languages
	for trace_expr, switch in zip(trace_exprs_val[idx], switches_val[idx]):
		trace_expr.print_constructor()
		switch.print_constructor()
		print(idx)
		print("")

	return trace_exprs_val[idx], switches_val[idx], accs[idx]