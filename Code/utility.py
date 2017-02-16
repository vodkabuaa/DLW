import numpy as np

class EZUtility(object):
	"""Calculation of Epstein-Zin utility for the DLW-model.

	Args:
		tree (obj 'TreeModel'):
		damage (obj 'Damage'):
		cost (obj 'Cost'):
		utility_tree (obj 'UtilityTree'): 
		eis (float): Elasticity of intertemporal substitution.
		ra (float): Risk-aversion.
		time_pref (float): Pure rate of time preference.

	Attributes:
		tree (obj 'TreeModel'):
		damage (obj 'Damage'):
		cost (obj 'Cost'):
		utility_tree (obj 'UtilityTree'): 
		eis (float): Elasticity of intertemporal substitution.
		ra (float): Risk-aversion.
		time_pref (float): Pure rate of time preference.
		r (float): Parameter rho from the DLW paper
		a (float): Parameter alpha in the DLW paper
		b (float): Parameter beta in the DLW paper

	"""

	def __init__(self, tree, damage, utility_tree, cost, eis=0.9, ra=7.0, time_pref=0.005):
		self.tree = tree
		self.damage = damage
		self.cost = cost
		self.utility_tree = utility_tree
		self.eis = eis
		self.ra = ra
		self.time_pref = time_pref
		self.r = 1.0 - 1.0/eis
		self.a = 1.0 - ra
		self.b = (1.0-time_pref)**utility_tree.subinterval_len

	def utility_function(self, m):
		pass

