import sys
sys.path.append("C:\\Users\\minori\\Desktop\\PBE_codes")
import re

alphaReg = re.compile(r'^[a-zA-Z]+$')
def isalpha(s):
    return alphaReg.match(s) is not None

def NumTok(string):
	node = re.findall(r"[0-9]+" , string)
	return node

def AlphaTok(string):
	node = re.findall(r"[a-zA-Z]+" , string)
	return node

def SpaceTok(string):
	node = [" "]
	return node

def PeriodTok(string):
	node = ["."]
	return node

def CommaTok(string):
	node = [","]
	return node

def LeftParenthesisTok(string):
	node = ["("]
	return node

def RightParenthesisTok(string):
	node = [")"]
	return node

def DQuoteTok(string):
	node = ["\""]
	return node

def SQuoteTok(string):
	node = ["'"]
	return node

def HyphenTok(string):
	node = ["-"]
	return node

def UBarTok(string):
	node = ["_"]
	return node

def SlashTok(string):
	node = ["/"]
	return node

def StartTok(string):
	node = string[0]
	return node

def EndTok(string):
	node = string[-1]
	return node

def EOFTok(string):
	node = [""]
	return node

def NoneTok(string):
	node = [None]
	return node

def MatchTok(string):
	node = [string]
	return node