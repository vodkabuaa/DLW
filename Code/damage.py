# -*- coding: utf-8 -*-

import numpy as np
from damage_simulation import DamageSimulation
from forcing import Forcing

class Damage(object):
	def __init__(
			self, tree, bau, ghg_start=400, force_simul=1.0,
			ghg_levels=[450, 650, 1000]):
		self.tree = tree
		self.bau = bau
		self.ghg_start = ghg_start
		self.ghg_levels = np.array(ghg_levels)
		self.force_simul = force_simul
		self.dnum = len(ghg_levels)
		self.cum_forcings = None
		self.d = None
		self.forcing = None
		self.damage_coefs = None

	def _damage_simulation(self, kwargs):
		if 'draws' not in kwargs.keys():
			kwargs['draws'] = 4000000

		damage_sim = DamageSimulation(self.tree, self.ghg_levels, kwargs)
		self.d = damage_sim.simulate()

	def _import_damages(self, loc="./data/simulated_damages.csv"):
		with open(loc, 'r') as f:
			d = np.loadtxt(f, delimiter=";")
		n = self.tree.num_final_states
		self.d = np.array([d[n*i:n*(i+1)] for i in range(0, self.dnum)])

	def _damage_interpolation(self):
		"""Create the interpolation coeffiecients used to calculate damages.
		"""
		self.damage_coefs = np.zeros((self.tree.num_final_states, self.tree.num_periods, self.dnum-1, self.dnum))
		amat = np.ones((self.tree.num_periods, self.dnum, self.dnum))
		bmat = np.ones((self.tree.num_periods, self.dnum))

		self.damage_coefs[:, :, -1,  -1] = self.d[-1, :, :]
		self.damage_coefs[:, :, -1,  -2] = (self.d[-2, :, :] - self.d[-1, :, :]) / self.emit_pct[-2]
		amat[:, 0, 0] = 2.0 * self.emit_pct[-2]
		amat[:, 1:, 0] = self.emit_pct[:-1]**2
		amat[:, 1:, 1] = self.emit_pct[:-1]

		for state in range(0, self.tree.num_final_states):
			bmat[:, 0] = self.damage_coefs[state, :, -1,  -2] * self.emit_pct[-2]
			bmat[:, 1:] = self.d[:-1, state, :].T
			self.damage_coefs[state, :, 0] = np.linalg.solve(amat, bmat)

	def forcing_based_mitigation(self, forcing, period):
		"""Calculation of mitigation based on forcing up to period.

		Args:
			forcing (float): Cumulative forcing up to node.
			period (int): Period of node.

		Returns:
			float: Mitigation.
		"""
		p = period - 1
		if forcing > self.cum_forcings[p][1]:
			weight_on_sim2 = (self.cum_forcings[p][2] - forcing) / (self.cum_forcings[p][2] - self.cum_forcings[p][1])
			weight_on_sim3 = 0
		elif forcing > self.cum_forcings[p][0] :
			weight_on_sim2 = (forcing - self.cum_forcings[p][0]) / (self.cum_forcings[p][1] - self.cum_forcings[p][0])
			weight_on_sim3 = (self.cum_forcings[p][1] - forcing) / (self.cum_forcings[p][1] - self.cum_forcings[p][0])
		else:
			weight_on_sim2 = 0
			weight_on_sim3 = 1.0 + (self.cum_forcings[p][0] - forcing) / self.cum_forcings[p][0]

		return weight_on_sim2 * self.emit_pct[1] + weight_on_sim3 * self.emit_pct[0]

	def average_mitigation(self, m, node):
		"""Assuming len(m) == num_periods OBS! not the same m as bob has"""
		period = self.tree.get_period(node)
		state = self.tree.get_state(node, period)
		period_len = self.tree.decision_times[1:period+1] - self.tree.decision_times[:period]
		bau_emissions = self.bau.emission_by_decisions[:period]
		total_emission = np.dot(bau_emissions, period_len)
		ave_mitigation = np.dot(m[:period], bau_emissions*period_len)
		return ave_mitigation / total_emission


	def damage_simulation_init(self, import_damages=True, **kwargs):
		"""Initializion of simulation of damages. Either import stored simulation
		of damages or simulate new values.

		Args:
			import_damages (bool): If program should import already stored values.
				Default is True.
			**kwargs: Arguments to initialize DamageSimulation class, in the
				case of import_damages = False. See DamageSimulation class for
				more info.

		"""
		if import_damages:
			self._import_damages()
		else:
			self._damage_simulation(kwargs)

	def forcing_init(self, **kwargs):
		"""Initialize Forcing object and cum_forcings used in calculating
		the mitigation up to a node.

		Args:
			**kwargs: Arguments to initialize Forcing object, see
				Forcing class for more info.

		"""
		bau_emission = self.bau.ghg - self.ghg_start
		self.emit_pct = 1.0 - (self.ghg_levels-self.ghg_start) / bau_emission
		self.cum_forcings = np.zeros((self.tree.num_periods, self.dnum))
		self.forcing = Forcing(self.tree, self.bau, kwargs)

		mitigation = np.ones((self.dnum, self.tree.num_decision_nodes)) * self.emit_pct[:, np.newaxis]
		path_ghg_levels = np.zeros((self.dnum, self.tree.num_periods+1))
		path_ghg_levels[0,:] = self.ghg_start

		for i in range(0, self.dnum):
			for n in range(1, self.tree.num_periods+1):
				node = self.tree.get_node(n, 0)
				self.cum_forcings[n-1, i] = self.forcing.forcing_at_node(mitigation[i], node, i)

	def damage_function(self, m, node):
		"""Calculate the damage at any given node, based on mitigation actions.

		Args:
			m (ndarray): Array of mitigation.
			node (int): The node for which damage is to be calculated.

		Returns:
			float: damage at node.

		"""
		if self.damage_coefs is None:
			self._damage_interpolation()

		if node == 0:
			return 0.0

		period = self.tree.get_period(node)
		worst_end_state, best_end_state = self.tree.reachable_end_states(node, period=period)

		ave_mitigation = self.forcing_based_mitigation(self.forcing.forcing_at_node(m, node), period)
		probs = self.tree.final_states_prob[worst_end_state:best_end_state+1]


		if ave_mitigation < self.emit_pct[1]:
			damage = (self.damage_coefs[worst_end_state:best_end_state+1, period-1, 1, 1] * ave_mitigation \
					 + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 1, 2]).sum()
		elif ave_mitigation < self.emit_pct[0]:
			damage = (probs * (self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 0]*ave_mitigation**2 \
					  + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 0]*ave_mitigation \
					  + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 1, 2])).sum()

		else:
			if np.any(self.d[worst_end_state:best_end_state+1, period-1, 0] > 0.00001):
				print ("something wierd")
			deriv = 2.0 * self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 0]*self.emit_pct[0] \
					+ self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 1]
			decay_scale = deriv / (self.d[worst_end_state:best_end_state+1, period-1, 0]*np.log(0.5))
			dist = ave_mitigation - self.emit_pct[0] + np.log(self.d[worst_end_state:best_end_state+1, period-1, 0]) \
				   / (np.log(0.5) * decay_scale)
			damage = probs * 0.5**(decay_scale*dist) * np.exp(-(avE_mitigation-self.emit_pct[0])**2/60.0)

		return damage / probs.sum()

if __name__ == "__main__":
	from tree import TreeModel
	from bau import BusinessAsUsual
	from utility_tree import UtilityTree

	t = TreeModel()
	ut = UtilityTree()
	bau_default_model = BusinessAsUsual()
	bau_default_model.bau_emissions_setup(t)


	df = Damage(tree=t, bau=bau_default_model)
	df.damage_simulation_init(
		import_damages=False, peak_temp=6.0, disaster_tail=18.0, tip_on=True,
		temp_map=1, temp_dist_params=None, pindyck_impact_k=4.5,
		pindyck_impact_theta=21341.0, pindyck_impact_displace=-0.0000746,
		maxh=100.0, cons_growth=0.015)
	df.forcing_init(
		sink_start=35.596, forcing_start=4.926, ghg_start=400, partition_interval=5,
		forcing_p1=0.13173, forcing_p2=0.607773, forcing_p3=315.3785, absorbtion_p1=0.94835,
		absorbtion_p2=0.741547, lsc_p1=285.6268, lsc_p2=0.88414)
