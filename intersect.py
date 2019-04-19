import sys
sys.path.append("C:\\Users\\minori\\Desktop\\PBE_codes")

import itertools
import numpy as np
import copy
import tools_pbe as tls

# Compare two expressions and sreturn 1 if two are identical and return 0 if not
def expr_comp(expr1, expr2):
	if expr1.id == expr2.id:
		if expr1.id == "SubStr" and expr1.Token == expr2.Token and expr1.num == expr2.num:
			return 1
		elif expr1.id == "ConstStr" and expr1.get_value() == expr2.get_value():
			return 1
		elif expr1.id == "FirstStr" and expr2.get_value() == expr2.get_value():
			return 1
		elif expr1.id == "MatchStr" and expr2.get_value() == expr2.get_value():
			return 1
		else:
			return 0
	else:
		return 0

# Count the number of common expressions included in Graphs for each expression in graphs1
def common_num_count(graphs1, Graphs):
	counts = list()
	for graph1 in graphs1:
		count = 0
		for graphs2 in Graphs:
			for graph2 in graphs2:
				if graphs1 is not graphs2:
					count = count + expr_comp(graph1.W[0], graph2.W[0])
		counts.append(count)
	return counts

def max_common_num_count(Graphs, max_num):
	common_num_list = list()
	for graphs in Graphs:
		common_num_list.append(common_num_count(graphs, Graphs))
	if max_num == "init":
		max_common_num = max([max(common_num_list[i]) for i in range(0, len(common_num_list))])
	else:
		for i in range(0, len(common_num_list)):
			for j in range(0, len(common_num_list[i])):
				if common_num_list[i][j] > max_num:
					common_num_list[i][j] = -1
		max_common_num = max([max(common_num_list[i]) for i in range(0, len(common_num_list))])
	return max_common_num

def make_common_num_list(Graphs, max_num):
	common_num_list = list()
	for graphs in Graphs:
		common_num_list.append(common_num_count(graphs, Graphs))
	for i in range(0, len(common_num_list)):
		for j in range(0, len(common_num_list[i])):
			if common_num_list[i][j] > max_num:
				common_num_list[i][j] = -2
	return common_num_list

# Return generalized expression (which has no String)
def generalize_expr(expression):
	if expression.id == "SubStr":
		gen_expr = tls.SubStr(None, expression.Token, expression.num)
	elif expression.id == "FirstStr":
		gen_expr = tls.FirstStr(None)
	elif expression.id == "ConstStr":
		gen_expr = tls.ConstStr(expression.Output)
	elif expression.id == "MatchStr":
		gen_expr = tls.MatchStr(None, expression.get_value())
	else:
		gen_expr = expression
	return gen_expr

def search_expr(Graphs, common_num_list, max_num):
	exprs = list()

	max_common_num = max([max(common_num_list[i]) for i in range(0, len(common_num_list))])
	if max_common_num == -2:
		return exprs
	else:
		for i in range(0, len(common_num_list)):
			for j in range(0, len(common_num_list[i])):
				if common_num_list[i][j] == max_common_num:
					selected = generalize_expr(Graphs[i][j].W[0])
					duplicity = 0
					if len(exprs) != 0:
						for expr in exprs:
							if expr_comp(expr, selected):
								duplicity = 1
					if duplicity == 0:
						exprs.append(selected)
		return exprs

def select_graph(expression, graphs, common_num_list):
	for graph, common_nums in zip(graphs, common_num_list):
		expression2 = generalize_expr(graph.W[0])
		if expr_comp(expression, expression2):
			return graph, common_nums
	return -1, -1

# Return 1 if same one is found, 0 if unique
def comparator_all(classes_list, classes1):
	for classes2 in classes_list:
		if comparator(classes1, classes2):
			return 1
	return 0

# Compare two groups of classes and return 1 if same, 0 if different
def comparator(classes1, classes2):
	classes1_c = copy.deepcopy(classes1)
	classes2_c = copy.deepcopy(classes2)
	if len(classes1_c) != len(classes2_c):
		return 0
	else:
		for i in range(0, len(classes1_c)):
			for j in range(0, len(classes2_c)):
				if classes1_c[i] == classes2_c[j]:
					classes1_c.pop(i)
					classes2_c.pop(j)
					if len(classes1_c)==0 and len(classes2_c)==0:
						return 1
					else:
						return comparator(classes1_c, classes2_c)
		return 0

def Make_partitions(Graphs, common_num_list, max_num=0):
	partitions =list()
	trace_exprs = list()
	expressions = search_expr(Graphs, common_num_list, max_num) # SubStr, FirstStr
	#for expression in expressions:
		#print(expression.get_value())
	if len(expressions) == 0:
		return -1, -1 # No corresponding expressions's been found
	else:
		for expression in expressions:
			selected_graphs = list()
			selected_graph = list()
			trace_expr = list()
			Graphs2 = list()
			common_num_list2 = list()
			for graphs, common_nums in zip(Graphs, common_num_list):
				graph, common_num = select_graph(expression, graphs, common_nums)
				if graph != -1:
					#graph.print_constructor()
					selected_graph.append(graph)
				else:
					Graphs2.append(graphs)
					common_num_list2.append(common_nums)
			selected_graphs.append(selected_graph) #selected graphs = [sub,sub,sub,sub,sub], Graphs2 = [none, none]
			trace_expr.append(expression)
			if len(Graphs2) == 0:
				if not comparator_all(trace_exprs, trace_expr):
					partitions.append(selected_graphs)
					trace_exprs.append(trace_expr)
			else:
				remainings, remain_exprs = Make_partitions(Graphs2, common_num_list2, max_num=max_num)
				if remainings == -1:
					pass
				else:
					for remain, remain_expr in zip(remainings, remain_exprs):
						selected_graphs2 = selected_graphs + remain
						trace_expr2 = trace_expr + remain_expr
						if not comparator_all(trace_exprs, trace_expr2):
							partitions.append(selected_graphs2)
							trace_exprs.append(trace_expr2)
		return partitions, trace_exprs

def check_contain(candidate, token_withidxs):
	assert len(candidate) > 0
	# ex.) regular_expr --> [AlphaTok(0), NumTok(0), SpaceTok(0), AlphaTok(1), NumTok(1)]
	for token_withidx in token_withidxs:
		for part in candidate:
			#print("CHECK: ", part.RegExpr, part.num, regular_expr[0])
			if part in token_withidx:
				pass
			else:
				return 0
	return 1

def search_match_regular_exprs(nodes_all):
	match = nodes_all[0]
	matches = list()

	# Find match tokens
	for nodes in nodes_all:
		match_set = set(match)
		nodes_set = set(nodes)
		match = list(match_set & nodes_set)
	print("(search_match_regular_exprs)", nodes_all[0], match)
	# make Match constructors
	for token in match:
		if not tls.Nonecheck(token):
			print("(search_match_regular_exprs)", token, type(token))
			regular_expr = [tls.Regular_expr("MatchTok", 1, matchtoken=token)]
			matches.append(tls.Match(regular_expr, 1))

	return matches

def make_adjacent(token_withidxs):

	return adjacents

def compare_adjacent(token_withidxs):
	if 1:
		return True
	else:
		return False

def search_remain(adjacent, token_withidxs):
	remain = list()
	for token_withidx in token_withidxs:
		if not token_withidx in adjacent:
			remain.append(token_withidx)
	return remain

def disjoint_adjacents(token_withidxs):
	# Product (Make all possible combinations)
	candidates = list()
	for i in range(2, len(token_withidxs)):
		candidates.extend(list(itertools.combinations(token_withidxs,i)))

	adjacents, remains = list(), list()
	for candidate in candidates:
		flag = True
		num = candidate[0].tid - 1
		for part in candidate:
			if part.tid - 1 == num:
				num = part.tid
			else:
				flag = False
		if flag:
			adjacents.append(candidate)
			remains.append(search_remain(candidate, token_withidxs))

	return adjacents, remains

def make_conj_wo_adjacent(token_withidxs):
	conjunctions = list()

	#Enumerate all token types
	tokens_dict = dict()
	for token_idx in token_withidxs:
		tokens_dict[token_idx.Token] = token_idx.num + 1

	matches = list()
	for token in tokens_dict:
		matches.append(tls.Match([token], tokens_dict[token]))

	conjunctions.append(tls.Conjunction(matches))

	return conjunctions

def make_conj_by_tokenidxs(token_withidxs):
	conjunctions = list()
	#Make conjunction without considering adjacent tokens
	conjunctions.extend(make_conj_wo_adjacent(token_withidxs))

	#Make conjunctions considering adjacent tokens
	adjacents, remains = disjoint_adjacents(token_withidxs)
	for adjacent, remain in zip(adjacents, remains):
		#Make new match to be added
		match = tls.Match(adjacent, 1)
		#Get conjunctions made by remains
		conjunctions_remain = make_conj_by_tokenidxs(remain)
		#Add new match
		for conjunction in conjunctions_remain:
			matches_remain = conjunction.Matches
			matches_new = list()
			flag = True
			for match_remain in matches_remain:
				if match == match_remain and flag:
					match_remain.num += 1
					flag = False
				matches_new.append(match_remain)
			if flag:
				matches_new.append(match)

			conjunction_new = tls.Conjunction(matches_new)
			if not conjunction_new in conjunctions:
				conjunctions.append(conjunction_new)
	return conjunctions

def search_common_regular_exprs(token_withidxs):
	# Search shortest tokens
	shortest_idx = np.argmin([len(v) for v in token_withidxs])

	# Product (Make all possible combinations)
	candidates = list()
	for i in range(0, len(token_withidxs[shortest_idx])):
		candidates.extend(list(itertools.combinations(token_withidxs[shortest_idx],i+1)))

	# Check every tokens whether they contain a regular expression, and if not, reject the regular expression
	chosens = list()
	for candidate in candidates: #candidate --> [AlphaTok(0), NumTok(0)]
		if check_contain(candidate, token_withidxs): # True if all input contain the regular expression
			chosens.append(candidate)

	# Find proper conjunctions
	conjunctions = list()
	for chosen in chosens:
		conjunctions.extend(make_conj_by_tokenidxs(chosen))

	# Return regular expressions which satisfy all of the cases
	return conjunctions #[Match([Al(0)],1), Match([Num(0)],1), Match([Al(0), Num(0)],1)]

def make_matches(partition):
	#Convert graph into regular expression
	token_withidxs = list()
	matches = list()
	for graph in partition:
		nodes = graph.eta_s
		token_withidxs.append(tls.make_token_withidx(nodes))

	#Search common matches
	matches.extend(search_common_regular_exprs(token_withidxs))
	#matches.extend(search_match_regular_exprs(nodes_all))
	return matches #[Match([Al(0)],1), Match([Num(0)],1), Match([Al(0), Num(0)],1)]

def make_conjunctions(partition):
	#Convert graph into regular expression
	token_withidxs = list()
	conjunctions = list()
	for graph in partition:
		nodes = graph.eta_s
		token_withidxs.append(tls.make_token_withidx(nodes))
	#Search common conjunctions
	conjunctions.extend(search_common_regular_exprs(token_withidxs))
	return conjunctions

def make_switches(partitions):
	# Make matches for each partition
	matches = list()
	for partition in partitions:
		matches.append(make_conjunctions(partition)) #[Match([Al(0)],1), Match([Num(0)],1), Match([Al(0), Num(0)],1)]
	# Product
	matches_prod = list(itertools.product(*matches))

	# Convert from tuple to list
	matches_prod_new = list()
	for matches in matches_prod:
		matches_prod_new.append(list(matches))

	# ex) ["A21 A22", "C231", "B10"], ["None"]
	return matches_prod_new #[[Match([Al(0)],1), Match([N(0)],1)], [Match([Num(0)],1), Match([N(0)], 1)], [Match([Al(0), Num(0)], 1), Match([N(0)],1)]]

def partitions_traceexprs_map(partitions, trace_exprs, switches):
	# Duplicate partitions
	partitions_new = list()
	trace_expr_new = list()
	for i in range(0, len(switches)):
		partitions_new.append(partitions)
		trace_expr_new.append(trace_exprs)

	return partitions_new, trace_expr_new

def check_disjoint(DAG, switch):
	#Check satisfuction of match for DAG
	yn = list()
	for match in switch:
		yn.append(match.Conditional(DAG.get_input()))

	#Return 1 if DAG is satisfied by only one match in switch, and 0 if not
	if sum(yn) == 1:
		return 1
	else:
		return 0

#Check classifier is disjoint
def validation(partitions, switch):
	#Make input list
	DAGs = list()
	for partition in partitions:
		DAGs.extend(partition)

	#Check disjoint for each input and return 0 if not disjoint, 1 if disjoint
	for DAG in DAGs:
		if check_disjoint(DAG, switch):
			pass
		else:
			return 0
	return 1

def Make_classifier(partitions_all, trace_exprs):
	#Make switches for all of partitions
	switches_all = list()
	for partitions in partitions_all:
		switches_all.append(make_switches(partitions))

	# Print all possible switches.
	"""
	print("SWITCHES: ")
	for switches in switches_all:
		print("----switch----")
		for switch in switches:
			for swi in switch:
				swi.print_constructor()
			print("")
		print("")
	"""

	#Map Match for each partitions
	partitions_all_map = list()
	trace_exprs_map = list()
	for partitions, trace_expr, switches in zip(partitions_all, trace_exprs, switches_all):
		# Switch can be multiple (ex. [Alpha / None], [Num / None], [Alpha&Num / None]), so partitions and traces should be mapped
		partitions_map, trace_expr_map = partitions_traceexprs_map(partitions, trace_expr, switches)
		partitions_all_map.append(partitions_map)
		trace_exprs_map.append(trace_expr_map)

	#Check validation of switch - partition pair and use only valid ones
	partitions_val, trace_exprs_val, switches_val = list(), list(), list()
	for _partitions_all, _trace_exprs, switches in zip(partitions_all_map, trace_exprs_map, switches_all):
		for partitions, trace_expr, switch in zip(_partitions_all, _trace_exprs, switches):
			if validation(partitions, switch):
				partitions_val.append(partitions)
				trace_exprs_val.append(trace_expr)
				switches_val.append(switch)

	print(len(partitions_val), len(trace_exprs_val), len(switches_val))

	return partitions_val, trace_exprs_val, switches_val

def INTERSECT(Graphs):
	#Make partitions
	max_common_num = max_common_num_count(Graphs, "init")

	partitions_all = list()
	trace_exprs = list()
	while max_common_num >= 0:
		common_num_list = make_common_num_list(Graphs, max_common_num)
		partitions, trace_expr = Make_partitions(Graphs, common_num_list, max_num=max_common_num)
		partitions_all.extend(partitions)
		trace_exprs.extend(trace_expr)
		max_common_num = max_common_num_count(Graphs, max_common_num-1)

	#Print all possible partitions.
	"""
	for partitions, trace_expr in zip(partitions_all, trace_exprs):
		print("PARTITION:")
		for partition, trace in zip(partitions, trace_expr):
			print("----partition----")
			trace.print_constructor()
			for graph in partition:
				graph.print_constructor()
				nodes = copy.deepcopy(graph.eta_s)
		print("")
	"""
	#Make classifiers
	partitions_val, trace_exprs_val, switches_val = Make_classifier(partitions_all, trace_exprs)

	#Print all domain-specific languages
	i = 0
	print("---ALL VALID DOMAIN-SPECIFIC LANGUAGES---")
	for partitions, trace_exprs, switches in zip(partitions_val, trace_exprs_val, switches_val):
		print("Domain-specific language {}: ".format(i))
		for partition, trace_expr, switch in zip(partitions, trace_exprs, switches):
			print("Conditional (IF):")
			switch.print_constructor()
			print("Trace expression (THEN): ")
			trace_expr.print_constructor()
			print("Following examples are handled: ")
			[i.print_constructor() for i in partition]
			print("")
		i+=1
		print("")

	return partitions_val, trace_exprs_val, switches_val