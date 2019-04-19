import sys
sys.path.append("C:\\Users\\minori\\Desktop\\PBE_codes")
import tokens as tk
import numpy as np
from functools import total_ordering
import math

class Token_withidx():
	def __init__(self, tid, Token, num, matchtoken=None):
		self.tid = tid
		self.Token = Token
		self.num = num
		self.matchtoken = matchtoken

	def __eq__(self, other):
		if not isinstance(other, Token_withidx):
			return NotImplemented
		return (self.tid, self.Token, self.num, self.matchtoken) == (other.tid, other.Token, other.num, other.matchtoken)

	def __lt__(self, other):
		if not isinstance(other, Token_withidx):
			return NotImplemented
		return (self.tid, self.Token, self.num, self.matchtoken) < (other.tid, other.Token, other.num, other.matchtoken)

def index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]

def get_indices(tokens, token_withidxs):
	Tokenname_list = ['AlphaTok', 'NumTok', 'SpaceTok', 'PeriodTok', 'CommaTok', 'LeftParenthesisTok', 'RightParenthesisTok', 
						'SQuoteTok', 'DQuoteTok', 'HyphenTok', 'UBarTok', 'SlashTok', 'NoneTok']

	for Tokenname in Tokenname_list:
		indices = index_multi(tokens, Tokenname)
		for n, idx in enumerate(indices):
			token_withidxs[idx].num = n
	return 0

# Specify regular expressions for input graph (MatchTok would not be created)
def make_token_withidx(nodes):
	tokens = list()
	token_withidx = list()
	for i, node in enumerate(nodes):
		token = identifyToken(node) #"AlphaToken"
		assert token != -1
		tokens.append(token)
		token_withidx.append(Token_withidx(i, token, -1))

	get_indices(tokens, token_withidx)

	return token_withidx

class Match():
	def __init__(self, TokenSeq, num):
		self.TokenSeq = TokenSeq
		self.num = num
		self.id = "Match"

	def __eq__(self, other):
		if not isinstance(other, Match):
			return NotImplemented
		return (self.TokenSeq, self.num) == (other.TokenSeq, other.num)

	def __lt__(self, other):
		if not isinstance(other, Match):
			return NotImplemented
		return (self.TokenSeq, self.num) < (other.TokenSeq, other.num)

	def Conditional(self, String):
		if self.get_tokenseq_len(String) >= self.num:
			return True
		else:
			return False

	def get_tokenseq_len(self, String):
		# Divede string into regular expression
		nodes = Makenode(String, [])

		count = self.tokenseq_check(nodes, 0)
		return count

	def tokenseq_check(self, nodes, count):
		assert len(self.TokenSeq) > 0
		if len(self.TokenSeq) > len(nodes):
			return count
		else:
			flag = [0 for i in range(0, len(self.TokenSeq))]
			for i in range(0, len(self.TokenSeq)):
				node_tok = identifyToken(nodes[i])
				if node_tok == self.TokenSeq[i]:
					flag[i] = 1
			if np.min(flag) == 1:
				count += 1
			nodes = nodes[1:]
			return self.tokenseq_check(nodes, count)

	def print_constructor(self):
		print("(Match)", self.__class__.__name__, "(", [i for i in self.TokenSeq], ",", self.num, ")")

	def return_constructor(self):
		TokenStr = ""
		for i in range(0, len(self.TokenSeq)):
			TokenStr += self.TokenSeq[i]
			TokenStr += ", "
		print(TokenStr)
		print(type(TokenStr))
		string = "(ATOM)" + self.__class__.__name__ + "(" + TokenStr + "," + str(self.num) + ")"
		return string


class OLD_Match():
	def __init__(self, TokenSeq, num):
		self.TokenSeq = TokenSeq
		self.num = num
		self.id = "Match"

	def __eq__(self, other):
		if not isinstance(other, Match):
			return NotImplemented
		return (self.TokenSeq, self.num) == (other.TokenSeq, other.num)

	def __lt__(self, other):
		if not isinstance(other, Match):
			return NotImplemented
		return (self.TokenSeq, self.num) < (other.TokenSeq, other.num)

	def Conditional(self, String):
		if self.get_tokenseq_len(String) >= self.num:
			return True
		else:
			return False

	def get_tokenseq_len(self, String):
		# Divede string into regular expression
		nodes = Makenode(String, [])
		reg = make_token_withidx(nodes)

		# Check if string contains all tokens in Match
		for regexpr in self.RegExprs:
			if regexpr.RegExpr == "MatchTok":
				if not Nonecheck(String) and regexpr.matchtoken in String:
					pass
				else:
					return 0
			else:
				if regexpr in reg:
					pass
				else:
					return 0
		return 1

	def print_constructor(self):
		print("(Match)", self.__class__.__name__, "(", [[i.RegExpr, i.num] for i in self.RegExprs], ",", self.num, ")")

class Conjunction():
	def __init__(self, Matches):
		self.Matches = Matches
		self.id = "Conjunction"

	def __eq__(self, other):
		if not isinstance(other, Conjunction):
			return NotImplemented
		return (self.Matches) == (other.Matches)

	def __lt__(self, other):
		if not isinstance(other, Conjunction):
			return NotImplemented
		return (self.Matches) < (other.Matches)

	def Conditional(self, String):
		for Match in self.Matches:
			if not Match.Conditional(String):
				return False
		return True

	def print_constructor(self):
		print("(Conjunction)", self.__class__.__name__)
		for Match in self.Matches:
			Match.print_constructor()

	def return_constructor(self):
		print("(Conjunction)", self.__class__.__name__)
		constructors = list()
		for Match in self.Matches:
			constructors.append(Match.return_constructor())
		return constructors


# Alpha(1) Num(0)
def makeMatch(RegExprs):

	return Match(RegExprs, num)

def getnode(String, Token):
	if Token == "EOFTok":
		node = tk.EOFTok(String)
	elif Token == "SpaceTok":
		node = tk.SpaceTok(String)
	elif Token == "PeriodTok":
		node = tk.PeriodTok(String)
	elif Token == "CommaTok":
		node = tk.CommaTok(String)
	elif Token == "LeftParenthesisTok":
		node = tk.LeftParenthesisTok(String)
	elif Token == "RightParenthesisTok":
		node = tk.RightParenthesisTok(String)
	elif Token == "DQuoteTok":
		node = tk.DQuoteTok(String)
	elif Token == "SQuoteTok":
		node = tk.SQuoteTok(String)
	elif Token == "HyphenTok":
		node = tk.HyphenTok(String)
	elif Token == "UBarTok":
		node = tk.UBarTok(String)
	elif Token == "SlashTok":
		node = tk.SlashTok(String)
	elif Token == "NoneTok":
		node = tk.NoneTok(String)
	elif Token == "NumTok":
		node = tk.NumTok(String)
	elif Token == "AlphaTok":
		node = tk.AlphaTok(String)
	else:
		print("getnode didn't match any token: ", String)
		return -1
	return node[0]

def Nonecheck(Input):
	if type(Input) != str and (Input == None or math.isnan(Input)):
		return True
	else:
		return False

def Makenode(String, Nodes):
	#print("(Node) String: ", String)
	Token = identifyToken(String)
	node = getnode(String, Token)
	#print("(Node) state: ", String, Token, node)
	if node == "":
		return Nodes
	elif Nonecheck(node):
		Nodes.append(node)
		return Nodes
	else:
		Nodes.append(node)
		String = String[len(node):]
		return Makenode(String, Nodes)

def identifyToken(String):
	if String == "":
		token = "EOFTok"
	elif Nonecheck(String):
		token = "NoneTok"
	else:
		firstchar = String[0]
		if firstchar.isdecimal():
			token = "NumTok"
		elif tk.isalpha(firstchar):
			token = "AlphaTok"
		elif firstchar == " ":
			token = "SpaceTok"
		elif firstchar == ".":
			token = "PeriodTok"
		elif firstchar == ",":
			token = "CommaTok"
		elif firstchar == "(":
			token = "LeftParenthesisTok"
		elif firstchar == ")":
			token = "RightParenthesisTok"
		elif firstchar == "\"":
			token = "DQuoteTok"
		elif firstchar == "'":
			token = "SQuoteTok"
		elif firstchar == "-":
			token = "HyphenTok"
		elif firstchar == "_":
			token = "UBarTok"
		elif firstchar == "/":
			token = "SlashTok"
		else:
			print("identifyToken couldn't find any token. Input: ", firstchar)
			return -1
	return token

@total_ordering
class SubStr():
	def __init__(self, String, Token, num):
		self.String = String
		self.Token = Token
		self.num = num
		self.id = "SubStr"

	def __eq__(self, other):
		if not isinstance(other, SubStr):
			return NotImplemented
		return (self.String, self.Token, self.num, self.id) == (other.String, other.Token, other.num, other.id)

	def __lt__(self, other):
		if not isinstance(other, SubStr):
			return NotImplemented
		return (self.String, self.Token, self.num, self.id) < (other.String, other.Token, other.num, other.id)

	def get_value(self):
		#print("string, token, num", self.String, self.Token, self.num)
		if Nonecheck(self.String):
			return ""
		else:
			#print(self.String, self.Token, self.num)
			try:
				x = self.getnode(self.String, self.Token)[self.num]
				return x
			except:
				return None

	def getnode(self, String, Token):
		if Token == "NumTok":
			node = tk.NumTok(String)
		elif Token == "AlphaTok":
			node = tk.AlphaTok(String)
		elif Token == "NoneTok":
			node = tk.NoneTok(String)
			print("NoneTok", node)
		elif Token == "StartTok":
			node = tk.StartTok(String)
		elif Token == "EndTok":
			node = tk.EndTok(String)
		else:
			return -1
		return node

	def print_constructor(self):
		print("(ATOM)", self.__class__.__name__, "(", self.String, ",", self.Token, ",", self.num, ")")

	def return_constructor(self):
		inp = self.String
		if self.String == None:
			inp = "None"

		string = "(ATOM)" + self.__class__.__name__ + "(" + inp + "," + self.Token + "," + str(self.num) + ")"
		return string

@total_ordering
class ConstStr():
	def __init__(self, Output):
		self.Output = Output
		self.id = "ConstStr"

	def __eq__(self, other):
		if not isinstance(other, ConstStr):
			return NotImplemented
		return (self.Output, self.id) == (other.Output, other.id)

	def __lt__(self, other):
		if not isinstance(other, ConstStr):
			return NotImplemented
		return (self.Output, self.id) < (other.Output, other.id)

	def get_value(self):
		return self.Output

	def print_constructor(self):
		print("(ATOM)", self.__class__.__name__, "(", self.get_value(), ")")

	def return_constructor(self):
		string = "(ATOM)" + self.__class__.__name__ + "(" + self.get_value() + ")"
		return string

@total_ordering
class FirstStr():
	def __init__(self, String):
		self.String = String
		self.id = "FirstStr"

	def __eq__(self, other):
		if not isinstance(other, FirstStr):
			return NotImplemented
		return (self.String, self.id) == (other.String, other.id)

	def __lt__(self, other):
		if not isinstance(other, FirstStr):
			return NotImplemented
		return (self.String, self.id) < (other.String, other.id)

	def get_value(self):
		#print("(FirstStr)", self.String)
		if Nonecheck(self.String):
			return self.String
		else:
			return self.String[0]

	def print_constructor(self):
		print("(ATOM)", self.__class__.__name__, "(", self.String, ")")

	def return_constructor(self):
		string = "(ATOM)" + self.__class__.__name__ + "(" + self.String + ")"
		return string


@total_ordering
class MatchStr():
	def __init__(self, String, Token):
		self.String = String
		self.Token = Token
		self.id = "MatchStr"

	def __eq__(self, other):
		if not isinstance(other, MatchStr):
			return NotImplemented
		return (self.String, self.Token, self.id) == (other.String, other.Token, other.id)

	def __lt__(self, other):
		if not isinstance(other, MatchStr):
			return NotImplemented
		return (self.String, self.Token, self.id) < (other.String, other.Token, other.id)

	def get_value(self):
		#print("string, token, num", self.String, self.Token, self.num)
		if Nonecheck(self.String):
			return self.String
		else:
			if self.Token in self.String:
				return self.Token

	def print_constructor(self):
		print("(ATOM)", self.__class__.__name__, "(", self.get_value(), ")")

	def return_constructor(self):
		string = "(ATOM)" + self.__class__.__name__ + "(" + self.get_value() + ")"
		return string


class DAG():
	def __init__(self, eta, eta_s, eta_t, xi, W):
		self.eta = eta # [eta_s, eta_t]
		self.eta_s = eta_s # ["C", "231"]
		self.eta_t = eta_t # ["C"]
		self.xi = xi # [(0, 0)]
		self.W = W # [SubStr(eta_s, tokne, num)]

	def estimated_output(self):
		return self.Concatenate()

	def print_constructor(self):
		#print("(DAG)", self.eta_s, self.eta_t, self.xi, self.W)
		if self.W[0].__class__.__name__ == "SubStr":
			print("(DAG:W)", self.W[0].__class__.__name__, "(", self.W[0].String, ",", self.W[0].Token, ",", self.W[0].num, ")")
		elif self.W[0].__class__.__name__ == "ConstStr":
			print("(DAG:W)", self.W[0].__class__.__name__, "(", self.W[0].get_value(), ")")
		elif self.W[0].__class__.__name__ == "FirstStr":
			print("(DAG:W)", self.W[0].__class__.__name__, "(", self.W[0].String, ")")
		elif self.W[0].__class__.__name__ == "MatchStr":
			print("(DAG:W)", self.W[0].__class__.__name__, "(", self.W[0].get_value(), ")")

	def Concatenate(self):
		expr = ""
		for w in self.W:
			node_est = w.get_value()
			expr += node_est
		return expr

	def get_input(self):
		if "" or None in self.eta_s:
			return None
		else:
			get_input = ""
			for part in self.eta_s:
				#print("part: ", part)
				get_input += part
			return get_input