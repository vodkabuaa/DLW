import numpy as np

class UtilityTree(object):

	def __init__(self, subinterval_len=5, decision_times=[0, 15, 45, 85, 185, 285, 385]):
		self.subinterval_len = subinterval_len
		self.decision_times = np.array(decision_times)
		self.information_times = self.decision_times[:-2]

		self.total_time = int(self.decision_times[-1] / self.subinterval_len) + 1
		self.utility_times = np.arange(0, self.decision_times[-1]+self.subinterval_len,
							 self.subinterval_len)
		
		self.utility_tree = dict.fromkeys(self.utility_times)
		i = 0
		for key in self.utility_times:
			self.utility_tree[key] = np.zeros(2**i)
			if key in self.information_times:
				i += 1


	def __repr__(self):
		return "Utility tree with {} nodes".format(len(self.utility_tree))

	def __iter__(self):
		"""
		Iterator for the UtilityTree class. Returns the items in 
		self.utility_tree sorted by their key in decending order.
		"""
		keys = self.utility_tree.keys()
		keys.sort(reverse=True)
		for key in keys:
			yield key, self.utility_tree[key]

	def is_decision_time(time_period):
		"""
		Checks if time_period is a decision time for mitigation, where
		time_period is the number of years since start.

		>>> is_decision_time(180)
		False
		>>> is_decision_time(185)
		True
		"""

		return time_period in self.decision_times

	def is_information_time(time_period):
		"""
		Checks if time_period is a information time for fragility, where
		time_period is the number of years since start.

		>>> is_information_time(185)
		True
		>>> is_information_time(285)
		False
		"""
		return time_period in self.information_times
