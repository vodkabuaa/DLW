import numpy as np

class Cost(object):
	"""Class to evaluate the cost curve for the DLW-model.

	Args:
		tree (obj 'TreeModel'): Provides the tree structure used.
		g (float): Intital scale of the cost function.
		a (float): Curvature of the cost function.
		join_price (float): Price at which the cost curve is extended.
		max_price (float): Price at which carbon dioxide can be removed from 
			atmosphere in unlimited scale.
		tech_const (float): Determines the degree of exogenous technological improvement 
			over time. A number of 1.0 implies 1 percent per yer lower cost.
		tech_scale (float): Determines the sensitivity of technological change 
			to previous mitigation. 
		cons_at_0 (float): Intital consumption. Default $30460bn based on US 2010 values.
		emit_at_0 (float): Initial GHG emission level.

	Attributes:
		tree (obj 'TreeModel'): Provides the tree structure used.
		g (float): Intital scale of the cost function.
		a (float): Curvature of the cost function.
		join_price (float): Price at which the cost curve is extended.
		max_price (float): Price at which carbon dioxide can be removed from 
			atmosphere in unlimited scale.
		tech_const (float): Determines the degree of exogenous technological improvement 
			over time. A number of 1.0 implies 1 percent per yer lower cost.
		tech_scale (float): Determines the sensitivity of technological change 
			to previous mitigation. 
		cbs_level (float): 
		cbs_deriv (float):
		cbs_b (float):
		cbs_k (float):
		cons_per_ton (float): Intitial consumption per ton GHG.
		cost_gradient (ndarray): Store the cost function gradient.

	"""

	def __init__(self, tree, g=92.08, a=3.413, join_price=2000.0, max_price=2500.0,
				tech_const=1.5, tech_scale=0.0, cons_at_0=30460.0):
		self.tree = tree
		self.bau = bau
		self.g = g
		self.a = a
		self.join_price = join_price
		self.max_price = max_price
		self.tech_const = tech_const
		self.tech_scale = tech_scale
		self.cbs_level = (join_price / (g + a))**(1.0 / (a - 1.0))
		self.cbs_deriv = self.cbs_level / (join_price * (a - 1.0))
		self.cbs_b = self.cbs_deriv * (max_price - join_price) / self.cbs_level
		self.cbs_k = self.cbs_level * (max_price - join_price)**self.cbs_b
		self.cons_per_ton = cons_at_0 / emit_at_0
		self.cost_gradient = np.zeros((tree.num_decision_nodes, tree.num_decision_nodes))

	def cost_by_state(self, node, mitigation, ave_mitigation):
		"""Calculates the mitigation cost by state.

		Args:
			node (int): Node in tree for which mitigation cost is calculated.
			mitigation (float): Current mitigation value
			ave_mitigation (float): Average mitigation per year up to this point.

		Returns:
			float: Cost by state (cbs)

		"""

		period = self.tree.get_period(node)
		years = self.tree.decision_times[period]

		tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation) / 100))**years
		if mitigation < self.cbs_level:
			cbs = self.g * (mitigation**self.a) * tech_term / self.cons_per_ton
		else:
			base_cbs = self.g * self.cbs_level**self.a
			extension = ((mitigation-self.cbs_level) * self.max_price 
						 - self.cbs_b*mitigation * (self.cbs_k/mitigation)**(1.0/self.cbs_b) / (self.cbs_b-1.0)
						 + self.cbs_b*self.cbs_level * (self.cbs_k/self.cbs_leve)**(1.0/self.cbs_b)/(self.cbs-1.0))
			cbs = (base_cbs + extension) * tech_term / self.cons_per_ton

		return cbs















