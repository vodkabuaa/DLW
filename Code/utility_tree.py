import numpy as np

"""
Author: Robert Litterman
Edit: Oscar Sjogren
"""

class UtilityTree(object):
	"""Utility tree class for the DLW-model.

	Attributes:
		subinterval_len (int): Lenght of subintervals between nodes where utility is calculated.
		decision_times (ndarray): List of times where decisions of mitigation is taken.
		information_times (ndarray): List of times where agent receives new information.
		? total_time (int): Total number arrays in utility tree
		utility_tree: (ndarray): 3D-array that represents the utility of agent for different states
								 at different time periods

	Args:
		subinterval_len (int): Lenght of subintervals between nodes where utility is calculated.
		decision_times (ndarray): List of times where decisions of mitigation is taken.

	"""

	def __init__(self, subinterval_len=5, decision_times=[0, 15, 45, 85, 185, 285, 385]):
		self.subinterval_len = subinterval_len
		self.decision_times = np.array(decision_times)
		self.information_times = self.decision_times[:-2]

		# NEEDED? self.total_time = int(self.decision_times[-1] / self.subinterval_len) + 1
		self.utility_times = np.arange(0, self.decision_times[-1]+self.subinterval_len,
							 self.subinterval_len)
		
		self.utility_tree = dict.fromkeys(self.utility_times)
		i = 0
		for key in self.utility_times:
			self.utility_tree[key] = np.zeros(2**i)
			if key in self.information_times:
				i += 1

	def __iter__(self):
		"""Generator which makes the UtilityTree class iteratable. Starts at the
		end of the tree.

		Yields:
			tuple: Key (int) and array (ndarray) of utility_tree

		"""
		keys = self.utility_tree.keys()
		keys.sort(reverse=True)
		for key in keys:
			yield key, self.utility_tree[key]

	def is_decision_time(time_period):
		"""Checks if time_period is a decision time for mitigation, where
		time_period is the number of years since start.

		Args:
			time_period (int): Time since the start year of the model.

		Returns:
			bool: True if time_period also is a decision time, else False.

		Example:
			>>> is_decision_time(180)
			False
			>>> is_decision_time(185)
			True

		"""
		return time_period in self.decision_times

	def is_information_time(time_period):
		"""Checks if time_period is a information time for fragility, where
		time_period is the number of years since start.

		Args:
			time_period (int): Time since the start year of the model.

		Returns:
			bool: True if time_period also is a information time, else False.

		Example:
			>>> is_information_time(185)
			True
			>>> is_information_time(285)
			False

		"""
		return time_period in self.information_times
