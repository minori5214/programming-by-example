import sys
sys.path.append("C:\\Users\\minori\\Desktop\\PBE_codes")
from statistics import mode

import numpy as np
import generate as gn
import intersect as ins
import ranking as rk
import tools_pbe as tls
import copy
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import csv

from collections import Counter

seed = 10

def random_forest_proc(x_train, x_test, y_train, y_test):	
	random_forest = RandomForestClassifier(n_estimators=150, random_state=0)
	random_forest.fit(x_train, y_train)
	acc_random_forest_tr = round(random_forest.score(x_train, y_train) * 100, 2)
	acc_random_forest_te = round(random_forest.score(x_test, y_test) * 100, 2)
	return acc_random_forest_tr, acc_random_forest_te

def random_forest_check(x_train, x_test, y_train, y_test, target_name, key=False):
	acc_tr, acc_te = random_forest_proc(x_train, x_test, y_train[target_name], y_test[target_name])
	print("acc: ", acc_tr, acc_te)
	if key:
		print(key)
		x_train_val = x_train.drop(key, axis=1)
		x_test_val = x_test.drop(key, axis=1)
		acc_tr_val, acc_te_val = random_forest_proc(x_train_val, x_test_val, y_train[target_name], y_test[target_name])
		print("POS(TRAIN, TEST), NEGA(TRAIN, TEST):", acc_tr, acc_te, acc_tr_val, acc_te_val)
		return acc_tr, acc_te, acc_tr_val, acc_te_val
	else:
		return acc_tr, acc_te, 0.0, 0.0

def Titanic_preprocess():
	train_df = pd.read_csv('./DATA/train.csv')
	test_df = pd.read_csv('./DATA/test.csv')
	combine = [train_df, test_df]

	train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace=True)
	test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)
	
	for dataset in combine:
		dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

	for dataset in combine:
		dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
			'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	freq_port = train_df.Embarked.dropna().mode()[0]
	for dataset in combine:
		dataset['Embarked'] = dataset['Embarked'].fillna('None')

	for dataset in combine:
		dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'None': -1} ).astype(int)

	for dataset in combine:
		dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
	for dataset in combine:
		dataset['Title'] = dataset['Title'].map(title_mapping)
		dataset['Title'] = dataset['Title'].fillna(0)
	
	train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)
	test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
	train_df = train_df.drop(["PassengerId", "Ticket", "Fare", "Name"], axis=1)
	test_df = test_df.drop(["PassengerId", "Ticket", "Fare", "Name"], axis=1).copy()

	return train_df, test_df

def feature_generation(data, trace_expr, switch):
	new_data = [dat for dat in data] #Copy data
	exceptions = list()
	count = 0
	valuelist = list() #for calculating mode

	# Replace values by data transformation
	for i in range(0, len(data)):
		for cond, trace in zip(switch, trace_expr):
			if cond.Conditional(data[i]):
				expr = copy.deepcopy(trace)
				expr.String = data[i]
				new_data[i] = expr.get_value()
				valuelist.append(new_data[i])
				count += 1
		if count == 0:
			exceptions.append(i)
		count = 0

	#Show edge case
	if len(exceptions) != 0:
		print("Value(s) not handled:")
		print(list(set([new_data[idx] for idx in exceptions])))
	mode = Counter(valuelist).most_common(1)[0][0]
	print("Alternative value: ", mode)

	#Replace exceptions (that cannot be handled by synthesized program) by mode
	for idx in exceptions:
		new_data[idx] = mode

	new_data = np.array(new_data)
	new_df = pd.DataFrame(data=new_data, columns=['new_data'], dtype=type(new_data[0]))
	return new_df['new_data']

def feature_generation_comb(x_train, x_test, trace_expr, switch):
	x_train_new = feature_generation(x_train, trace_expr, switch)
	x_test_new = feature_generation(x_test, trace_expr, switch)

	return x_train_new, x_test_new

def data_transformation(Name, Inputs, Outputs, x_train, x_test, y_train, y_test, target_name="Survived", seed=0, flashfill=False):
	Graphs = list()
	count = 0
	for _input, _output in zip(Inputs, Outputs):
		Graphs.append(gn.GENERATE(_input, _output))
		count = count + len(Graphs[-1])
	print("length of demos: ", len(Graphs))
	print("length of expressions: ", count)
	for graph in Graphs:
		print(len(graph))

	aaa = tls.SubStr("aaa", "AlphaTok", 0)
	print(aaa.get_value())
	bbb = tls.SubStr(None, "AlphaTok", 0)

	partitions_val, trace_exprs_val, switches_val = ins.INTERSECT(Graphs)

	fx_train = x_train[Name]
	fx_test = x_test[Name]

	if not flashfill:
		trace_expr, switch, max_acc = rk.RANKING(trace_exprs_val, switches_val, fx_train, y_train, target_name, seed=seed)
		print("(RANKING) MAX ACC: ", max_acc)

		new_fx_train, new_fx_test = feature_generation_comb(fx_train, fx_test, trace_expr, switch)
		return new_fx_train, new_fx_test, trace_expr, switch
	else:
		trace_expr, switch = rk.RANKING_FLASHFILL(trace_exprs_val, switches_val)
		new_fx_train, new_fx_test = feature_generation_comb(fx_train, fx_test, trace_expr, switch)
		return new_fx_train, new_fx_test, trace_expr, switch

def flashfill(Inputs, Outputs):
	Graphs = list()
	count = 0

	for _input, _output in zip(Inputs, Outputs):
		Graphs.append(gn.GENERATE(_input, _output))
		count = count + len(Graphs[-1])
	print("Total number of expressions: ", count)
	for i, graph in enumerate(Graphs):
		print("Number of graphs in Ex. {0}: {1}".format(i, len(graph)))
	print("")

	partitions_val, trace_exprs_val, switches_val = ins.INTERSECT(Graphs)

	trace_expr, switch = rk.RANKING_FLASHFILL(trace_exprs_val, switches_val)
	return trace_expr, switch

def plot_feature_importance(X, y, name):
	# Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=150,
	                              random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]
	labels = [X.columns.values[indices[i]] for i in range(0, len(indices))]
	print("Features: ", labels)

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), labels)
	plt.xlim([-1, X.shape[1]])
	plt.ylim([0.0, 0.5])
	plt.savefig('figure_{}.png'.format(name))


def start_process(Inputs, Outputs):
	for i in range(0, len(Outputs)):
		if Outputs[i] == None:
			Outputs[i] = "None"
	print("Inputs: ", Inputs)
	print("Outputs: ", Outputs)

	"""
	Inputs = ["C231", "B10", None, "A21 A22", "D"]
	Outputs = ["C", "B", "None", "A", "D"]
	print("Inputs: ", Inputs)
	print("Outputs: ", Outputs)
	"""

	train_df, _ = Titanic_preprocess()
	target_name = "Survived"
	np.random.seed(seed=seed)
	train_df = train_df.reindex(np.random.permutation(train_df.index))

	X = train_df.drop([target_name], axis=1)
	y = train_df[target_name]

	column_names = X.columns.values
	print(column_names)
	name_dict = dict()
	for i in range(0, len(column_names)):
		name_dict[i] = column_names[i]
	print(name_dict)

	num = int(len(train_df)/10)
	idx_s, idx_e = 0, num
	x_train	= np.vstack([X.values[0:idx_s], X.values[idx_e:]])
	x_test	= X.values[idx_s:idx_e]
	y_train	= np.vstack((y.values[0:idx_s].reshape(-1,1), y.values[idx_e:].reshape(-1,1)))
	y_test	= y.values[idx_s:idx_e]


	x_train = pd.DataFrame(x_train, columns=column_names)
	x_test = pd.DataFrame(x_test, columns=column_names)
	y_train = pd.DataFrame(y_train, columns=[target_name])
	y_test = pd.DataFrame(y_test, columns=[target_name])

	FG_train = x_train["Cabin"].copy()
	FG_test = x_test["Cabin"].copy()

	Names = ["Name", "Embarked", "Sex", "Cabin"]

	Name = Names[3]
	#Inputs = ["C231", "B10", None, "A21 A22", "D"]
	#Outputs = ["C", "B", "None", "A", "D"]
	#Inputs = ["C231 C15"]
	#Outputs = ["C"]
	#Inputs = ["C231", None, "C10"]
	#Outputs = ["C", "None", "C"]
	x_train[Name] = FG_train
	x_test[Name] = FG_test
	new_fx_train, new_fx_test, trace_exprs, switches = \
		data_transformation(Name, Inputs, Outputs, x_train, x_test, y_train, y_test, seed=seed, flashfill=True)
	x_train[Name] = new_fx_train
	x_test[Name] = new_fx_test
	print(np.array(new_fx_train)[0:100], len(np.array(new_fx_train)))
	print(np.array(new_fx_test)[0:], len(np.array(new_fx_test)))
	x_train, x_test, combine = rk.data_category_to_num(x_train, Name, x_test=x_test)

	check_names = [Names[3]]
	ACC_TR, ACC_TE, ACC_TR_VAL, ACC_TE_VAL = random_forest_check(x_train, x_test, y_train, y_test, target_name)#, key=check_names)
	#plot_feature_importance(x_train, y_train, 'train')
	#plot_feature_importance(x_test, y_test, 'test')

	#Print selected domain-specific languages
	for trace, swi in zip(trace_exprs, switches):
		trace.print_constructor()
		swi.print_constructor()
	print("")

	return trace_exprs, switches

def programming_by_example(Inputs, Outputs, synthesizer=flashfill):
	for i in range(0, len(Outputs)):
		if Outputs[i] == None:
			Outputs[i] = "None"
	print("Inputs: ", Inputs)
	print("Outputs: ", Outputs, "\n")

	if synthesizer==flashfill:
		trace_expr, switch = flashfill(Inputs, Outputs)
	else:
		raise NotImplementedError()

	#Print selected domain-specific languages
	print("(PBE) ---ADOPTED DOMAIN-SPECIFIC LANGUAGE---")
	for trace, swi in zip(trace_expr, switch):
		print("Conditional (IF):")
		swi.print_constructor()
		print("Trace expression (THEN): ")
		trace.print_constructor()
		print("")

	return trace_expr, switch

if __name__ == "__main__":
	train_df, _ = Titanic_preprocess()
	target_name = "Survived"
	for seed in range(0, 10):
		np.random.seed(seed=seed)
		train_df = train_df.reindex(np.random.permutation(train_df.index))

		X = train_df.drop([target_name], axis=1)
		y = train_df[target_name]

		column_names = X.columns.values
		print(column_names)
		name_dict = dict()
		for i in range(0, len(column_names)):
			name_dict[i] = column_names[i]
		print(name_dict)

		print(type(X.values))
		print(type(y.values))

		num = int(len(train_df)/10)
		print("num: ", num)
		ACC_TRS, ACC_TES, ACC_TRS_VAL, ACC_TES_VAL = list(), list(), list(), list()
		ACC_TRS_PRE, ACC_TES_PRE, ACC_TRS_VAL_PRE, ACC_TES_VAL_PRE = list(), list(), list(), list()
		ACC_TRS_N, ACC_TES_N, ACC_TRS_VAL_N, ACC_TES_VAL_N = list(), list(), list(), list()
		SCORESs_TR, SCORESs_TE = list(), list()

		trace_exprs_s, switches_s = list(), list()
		trace_exprs_pre_s, switches_pre_s = list(), list()

		for i in range(0, 10):
			idx_s, idx_e = i * num, (i+1) * num
			x_train	= np.vstack([X.values[0:idx_s], X.values[idx_e:]])
			x_test	= X.values[idx_s:idx_e]
			y_train	= np.vstack((y.values[0:idx_s].reshape(-1,1), y.values[idx_e:].reshape(-1,1)))
			y_test	= y.values[idx_s:idx_e]


			x_train = pd.DataFrame(x_train, columns=column_names)
			x_test = pd.DataFrame(x_test, columns=column_names)
			y_train = pd.DataFrame(y_train, columns=[target_name])
			y_test = pd.DataFrame(y_test, columns=[target_name])

			FG_train = x_train["Cabin"].copy()
			FG_test = x_test["Cabin"].copy()

			Names = ["Name", "Embarked", "Sex", "Cabin"]
			trace_exprs = [None for i in range(0, 4)]
			switches = [None for i in range(0, 4)]
			"""
			Name = Names[0]
			Inputs = ["Moran, Mr. James", "Masselmani, Mrs. Fatima"]
			Outputs = ["Mr", "Mrs"]
			new_fx_train, new_fx_test, trace_exprs[0], switches[0] = \
				data_transformation(Name, Inputs, Outputs, x_train, x_test, y_train, y_test, flashfill=True)
			x_train[Name] = new_fx_train
			x_test[Name] = new_fx_test
			print(np.array(new_fx_train)[0:20], len(np.array(new_fx_train)))
			print(np.array(new_fx_test)[0:20], len(np.array(new_fx_test)))
			x_train, x_test, combine = rk.data_category_to_num(x_train, Name, x_test=x_test)
			"""

			Name = Names[3]
			#Inputs = ["C231", "B10", None, None, "A21 A22", "C10"]
			#Outputs = ["C", "B", "None", "None", "A", "C"]
			Inputs = ["C231 C15"]
			Outputs = ["C"]
			#Inputs = ["C231", None, "C10"]
			#Outputs = ["C", "None", "C"]
			x_train[Name] = FG_train
			x_test[Name] = FG_test
			new_fx_train, new_fx_test, trace_exprs[2], switches[2] = \
				data_transformation(Name, Inputs, Outputs, x_train, x_test, y_train, y_test, seed=seed, flashfill=False)
			x_train[Name] = new_fx_train
			x_test[Name] = new_fx_test
			print(np.array(new_fx_train)[0:20], len(np.array(new_fx_train)))
			print(np.array(new_fx_test)[0:20], len(np.array(new_fx_test)))
			x_train, x_test, combine = rk.data_category_to_num(x_train, Name, x_test=x_test)

			check_names = [Names[3]]
			ACC_TR, ACC_TE, ACC_TR_VAL, ACC_TE_VAL = random_forest_check(x_train, x_test, y_train, y_test, target_name)#, key=check_names)
			#plot_feature_importance(x_train, y_train, 'train')
			#plot_feature_importance(x_test, y_test, 'test')

			Name = Names[3]
			#Inputs = ["C231", "B10", None, None, "A21 A22", "C10"]
			#Outputs = ["C", "B", "None", "None", "A", "C"]
			Inputs = ["C231 C15", None]
			Outputs = ["C", "None"]
			#Inputs = ["C231", None, "C10"]
			#Outputs = ["C", "None", "C"]
			x_train[Name] = FG_train
			x_test[Name] = FG_test
			new_fx_train, new_fx_test, trace_exprs_pre, switches_pre = \
				data_transformation(Name, Inputs, Outputs, x_train, x_test, y_train, y_test, seed=seed, flashfill=True)
			x_train[Name] = new_fx_train
			x_test[Name] = new_fx_test
			print(np.array(new_fx_train)[0:20], len(np.array(new_fx_train)))
			print(np.array(new_fx_test)[0:20], len(np.array(new_fx_test)))
			x_train, x_test, combine = rk.data_category_to_num(x_train, Name, x_test=x_test)

			check_names = [Names[3]]
			ACC_TR_PRE, ACC_TE_PRE, ACC_TR_VAL_PRE, ACC_TE_VAL_PRE = random_forest_check(x_train, x_test, y_train, y_test, target_name)#, key=check_names)
			#plot_feature_importance(x_train, y_train, 'train')
			#plot_feature_importance(x_test, y_test, 'test')


			"""
			#Previous Method
			check_names = [Names[3]]
			Name = Names[3]
			#Inputs = ["C231", "B10", None, None, "A21 A22", "C10"]
			#Outputs = ["C", "B", "None", "None", "A", "C"]
			#Inputs = ["C231 C15", None, "C10 C11"]
			#Outputs = ["C", "None", "C"]
			Inputs = ["C231", None, "C10"]
			Outputs = ["C", "None", "C"]
			x_train[Name] = FG_train
			x_test[Name] = FG_test
			new_fx_train, new_fx_test, trace_exprs_pre, switches_pre = \
				data_transformation(Name, Inputs, Outputs, x_train, x_test, y_train, y_test, flashfill=True)
			SCORES_TR, SCORES_TE = list(), list()
			for new_fx_train, new_fx_test in zip(new_fx_trains, new_fx_tests):
				x_train[Name] = new_fx_train
				x_test[Name] = new_fx_test
				print(np.array(new_fx_train)[0:20], len(np.array(new_fx_train)))
				print(np.array(new_fx_test)[0:20], len(np.array(new_fx_test)))
				x_train, x_test, combine = rk.data_category_to_num(x_train, Name, x_test=x_test)
				SCORE_TR, SCORE_TE, _, _ = random_forest_check(x_train, x_test, y_train, y_test, target_name, key=check_names)
				SCORES_TR.append(SCORE_TR)
				SCORES_TE.append(SCORE_TE)
			"""

			Inputs = ["C231", None, "C10", "D20"]
			Outputs = ["C", "None", "C", "C"]
			x_train[Name] = FG_train
			x_test[Name] = FG_test
			new_fx_train, new_fx_test, trace_exprs[3], switches[3] = data_transformation(Name, Inputs, Outputs, x_train, x_test, y_train, y_test, seed=seed)
			x_train[Name] = new_fx_train
			x_test[Name] = new_fx_test
			print(np.array(new_fx_train)[0:20], len(np.array(new_fx_train)))
			print(np.array(new_fx_test)[0:20], len(np.array(new_fx_test)))
			x_train, x_test, combine = rk.data_category_to_num(x_train, Name, x_test=x_test)

			check_names = [Names[3]]
			ACC_TR_N, ACC_TE_N, ACC_TR_VAL_N, ACC_TE_VAL_N = random_forest_check(x_train, x_test, y_train, y_test, target_name)#, key=check_names)

			ACC_TRS.append(ACC_TR)
			ACC_TES.append(ACC_TE)
			ACC_TRS_VAL.append(ACC_TR_VAL)
			ACC_TES_VAL.append(ACC_TE_VAL)

			ACC_TRS_PRE.append(ACC_TR_PRE)
			ACC_TES_PRE.append(ACC_TE_PRE)
			ACC_TRS_VAL_PRE.append(ACC_TR_VAL_PRE)
			ACC_TES_VAL_PRE.append(ACC_TE_VAL_PRE)

			ACC_TRS_N.append(ACC_TR_N)
			ACC_TES_N.append(ACC_TE_N)
			ACC_TRS_VAL_N.append(ACC_TR_VAL_N)
			ACC_TES_VAL_N.append(ACC_TE_VAL_N)

			#SCORESs_TR.append(SCORES_TR)
			#SCORESs_TE.append(SCORES_TE)

			trace_exprs_s.append(trace_exprs)
			switches_s.append(switches)
			trace_exprs_pre_s.append(trace_exprs_pre)
			switches_pre_s.append(switches_pre)

		for i in range(0, 10):

			print("POS_PROPOSED(TRAIN)               ", ACC_TRS[i])
			print("POS_PROPOSED(TEST)                ", ACC_TES[i])
			print("NEGA_PROPOSED(TRAIN)              ", ACC_TRS_VAL[i])
			print("NEGA_PROPOSED(TEST)               ", ACC_TES_VAL[i])
			print("")
			print("POS_PROPOSED NAIVE(TRAIN)         ", ACC_TRS_N[i])
			print("POS_PROPOSED NAIVE(TEST)          ", ACC_TES_N[i])
			print("NEGA_PROPOSED NAIVE(TRAIN)        ", ACC_TRS_VAL_N[i])
			print("NEGA_PROPOSED NAIVE(TEST)         ", ACC_TES_VAL_N[i])
			print("")
			print("POS_FLASHFILL(TRAIN)               ", ACC_TRS_PRE[i])
			print("POS_FLASHFILL(TEST)                ", ACC_TES_PRE[i])
			print("NEGA_FLASHFILL(TRAIN)              ", ACC_TRS_VAL_PRE[i])
			print("NEGA_FLASHFILL(TEST)               ", ACC_TES_VAL_PRE[i])
			print("")
			#print("FLASHFILL MEAN, BEST, WORST(TRAIN)", np.mean(SCORESs_TR[i]), np.max(SCORESs_TR[i]), np.min(SCORESs_TR[i]))
			#print("FLASHFILL MEAN, BEST, WORST(TEST) ", np.mean(SCORESs_TE[i]), np.max(SCORESs_TE[i]), np.min(SCORESs_TE[i]))

			#idx_tr = np.argmax(SCORESs_TR[i])
			#idx_te = np.argmax(SCORESs_TE[i])

			#Print selected domain-specific languages
			for trace_expr, switch in zip(trace_exprs_s[i], switches_s[i]):
				for trace, swi in zip(trace_expr, switch):
					trace.print_constructor()
					swi.print_constructor()
				print("")

			"""
			#Print selected domain-specific languages
			for trace_expr, switch in zip(trace_exprs_pre_s[i][idx_tr], switches_pre_s[i][idx_tr]):
				trace_expr.print_constructor()
				switch.print_constructor()
			print("")
			"""
			#Print selected domain-specific languages
			for trace_expr, switch in zip(trace_exprs_pre_s[i], switches_pre_s[i]):
				trace_expr.print_constructor()
				switch.print_constructor()
			print("")


		print("FINAL RESULTS: ")

		print("POS_PROPOSED(TRAIN)               ", np.mean(ACC_TRS), np.max(ACC_TRS), np.min(ACC_TRS))
		print("POS_PROPOSED(TEST)                ", np.mean(ACC_TES), np.max(ACC_TES), np.min(ACC_TES))
		print("NEGA_PROPOSED(TRAIN)              ", np.mean(ACC_TRS_VAL), np.max(ACC_TRS_VAL), np.min(ACC_TRS_VAL))
		print("NEGA_PROPOSED(TEST)               ", np.mean(ACC_TES_VAL), np.max(ACC_TES_VAL), np.min(ACC_TES_VAL))
		print("")
		print("POS_PROPOSED NAIVE(TRAIN)         ", np.mean(ACC_TRS_N), np.max(ACC_TRS_N), np.min(ACC_TRS_N))
		print("POS_PROPOSED NAIVE(TEST)          ", np.mean(ACC_TES_N), np.max(ACC_TES_N), np.min(ACC_TES_N))
		print("NEGA_PROPOSED NAIVE(TRAIN)        ", np.mean(ACC_TRS_VAL_N), np.max(ACC_TRS_VAL_N), np.min(ACC_TRS_VAL_N))
		print("NEGA_PROPOSED NAIVE(TEST)         ", np.mean(ACC_TES_VAL_N), np.max(ACC_TES_VAL_N), np.min(ACC_TES_VAL_N))
		print("")
		print("POS_FLASHFILL(TRAIN)               ", np.mean(ACC_TRS_PRE), np.max(ACC_TRS_PRE), np.min(ACC_TRS_PRE))
		print("POS_FLASHFILL(TEST)                ", np.mean(ACC_TES_PRE), np.max(ACC_TES_PRE), np.min(ACC_TES_PRE))
		print("NEGA_FLASHFILL(TRAIN)              ", np.mean(ACC_TRS_VAL_PRE), np.max(ACC_TRS_VAL_PRE), np.min(ACC_TRS_VAL_PRE))
		print("NEGA_FLASHFILL(TEST)               ", np.mean(ACC_TES_VAL_PRE), np.max(ACC_TES_VAL_PRE), np.min(ACC_TES_VAL_PRE))
		print("")
		#print("FLASHFILL MEAN, BEST, WORST(TRAIN)", np.mean(SCORESs_TR), np.max(SCORESs_TR), np.min(SCORESs_TR))
		#print("FLASHFILL MEAN, BEST, WORST(TEST) ", np.mean(SCORESs_TE), np.max(SCORESs_TE), np.min(SCORESs_TE))

		STACK = np.vstack((ACC_TRS, ACC_TES))
		STACK2 = np.vstack((ACC_TRS_PRE, ACC_TES_PRE))
		STACK3 = np.vstack((ACC_TRS_N, ACC_TES_N))
		STACK_VAL = np.vstack((ACC_TRS_VAL, ACC_TES_VAL))

		STACK = np.vstack((STACK, STACK2))
		STACK = np.vstack((STACK, STACK3))
		STACK = np.vstack((STACK, STACK_VAL))

		STACK = STACK.transpose()
		print(STACK.shape)

		with open('ranking_train_with.csv', 'a') as f:
			 writer = csv.writer(f, lineterminator='\n')
			 writer.writerows(STACK)