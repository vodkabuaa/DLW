import numpy as np
import multiprocessing as mp
from functools import partial
import copy_reg
import types

""" 
Author: Robert Litterman
Edit: Oscar Sjogren
"""

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class DamageSimulation(object):
    """Simulation of damages for the DLW-model.

    Args:
        tree (obj: 'TreeModel'): Provides the tree structure used.
        ghg_levels (ndarray or list): End GHG levels for each path.

    Attributes:
        tree (obj: 'TreeModel'): Provides the tree structure used.
        ghg_levels (ndarray or list): End GHG levels for each path.
        peak_temp (float): Determines the probability of a tipping point.
        disaster_tail (float): Curvature of the cost function.
        tip_on (bool): Flag that turns tipping points on or off.
        temp_map (int): The mapping from GHG to temperature.
            0 implies Pindyck displace gamma
            1 implies Wagner-Weitzman normal
            2 implies Roe-Baker
            3 implies user-defined normal 
            4 implies user-defined gamma
        temp_dist_params (ndarray or list): If temp_map is either 3 or 4, user needs
            to define the distribution parameters. 
        pindyck_impact_k (float): Shape parameter from Pyndyck damage function.
        pindyck_impact_theta (float): Scale parameter from Pyndyck damage function.
        pindyck_impact_displace (float): Displacement parameter from Pyndyck damage function.
        maxh (float): Time paramter from Pindyck which indicates the time it takes for temp to get half 
            way to its max value for a given level of ghg.
        cons_growth (float): Yearly growth in consumption.
        ghg_levels (list): Paths of GHG.
        d (ndarray): 3D-array with simulated damages for different GHG paths.

    """
  
    def __init__(self, tree, ghg_levels, kwargs):
        for key, val in kwargs.iteritems():
            setattr(self, key, val)
        self.tree = tree
        self.ghg_levels = ghg_levels
        self.d = None

    def _write_to_file(self):
        import os
        cwd = os.getcwd()
        if not os.path.exists('data'):
            os.makedirs('data')
        d = os.path.join(cwd, os.path.join('data','simulated_damages.csv'))

        with open(d, 'w') as f:
            for slice_2d in self.d:
                np.savetxt(f, slice_2d, delimiter=";")
                f.write('\n')

    def _gamma_array(self, shape, rate, dimension):
        return np.random.gamma(shape, 1.0/rate, dimension)

    def _normal_array(self, mean, stdev, dimension):
        return np.random.normal(mean, stdev, dimension)

    def _uniform_array(self, dimension):
        return np.random.random(dimension)

    def _sort_array(self, array):
        return array[array[:, self.tree.num_periods-1].argsort()]

    def _normal_simulation(self):
        """Draw random samples from normal distribution for mapping GHG to temperature for 
        user-defined distribution parameters.
        
        Args:
            draws (int): Number of draws from distribution.

        Returns: 
            ndarray

        """
        assert self.temp_dist_params and len(self.temp_dist_params) == 2, "Normal distribution needs 2 parameters."

        ave, std = temp_dist_params
        n = len(ave)
        temperature = np.array([self._normal_array(ave[i],std[i], self.draws) for i in range(0, n)])
        return np.exp(temperature)

    def _gamma_simulation(self):
        """Draw random samples from gamma distribution for mapping GHG to temperature for 
        user-defined distribution parameters.
        
        Args:
            draws (int): Number of draws from distribution.

        Returns: 
            ndarray

        """
        assert self.temp_dist_params and len(self.temp_dist_params) == 3, "Gamma distribution needs 3 parameters."

        k, theta, displace = temp_dist_params
        n = len(k)
        return np.array([self._gamma_array(k[i], theta[i], self.draws) 
                         + displace[i] for i in range(0, n)])

    def _pindyck_simulation(self):
        """Draw random samples for mapping GHG to temperature based on Pindyck.

        Args:
            draws (int): Number of draws from distribution.

        Returns: 
            ndarray

        """
        pindyck_temp_k = [2.81, 4.6134, 6.14]
        pindyck_temp_theta = [1.6667, 1.5974, 1.53139]
        pindyck_temp_displace = [-0.25, -0.5, -1.0]
        return np.array([self._gamma_array(pindyck_temp_k[i], pindyck_temp_theta[i], self.draws) 
                         + pindyck_temp_displace[i] for i in range(0, 3)])

    def _ww_simulation(self):
        """Draw random samples for mapping GHG to temperature based on Wagner-Weitzman.

        Args:
            draws (int): Number of draws from distribution.

        Returns: 
            ndarray

        """
        ww_temp_ave = [0.573, 1.148, 1.563]
        ww_temp_stddev = [0.462, 0.441, 0.432]
        temperature = np.array([self._normal_array(ww_temp_ave[i], ww_temp_stddev[i], self.draws) 
                                for i in range(0, 3)])
        return np.exp(temperature)

    def _rb_simulation(self):
        """Draw random samples for mapping GHG to temperature based on Roe-Baker.

        Args:
            draws (int): Number of draws from distribution.

        Returns: 
            ndarray

        """        
        rb_fbar = [0.75233, 0.844652, 0.858332]
        rb_sigf = [0.049921, 0.033055, 0.042408]
        rb_theta = [2.304627, 3.333599, 2.356967]
        temperature = np.array([self._normal_array(rb_fbar[i], rb_sigf[i], self.draws) 
                         for i in range(0, 3)])
        return np.maximum(0.0, (1.0 / (1.0 - temperature)) - np.array(rb_theta)[:, np.newaxis])

    def _pindyck_impact_simulation(self):
        """Pindyck gamma distribution mapping temperature into damages.

        Args:
            draws (int): Number of draws from distribution.
            
        Returns:
            ndarray

        """
        impact = self._gamma_array(self.pindyck_impact_k, self.pindyck_impact_theta, self.draws) + \
                 self.pindyck_impact_displace 
        return impact

    def _disaster_simulation(self):
        """Simulating disaster random variable, allowing for a tipping point to occur
        with a given probability, leading to a disaster and a "disaster_tail" impact on consumption.

        Args:
            draws (int): Number of draws from distribution.

        Returns:
            ndarray

        """
        disaster = self._uniform_array((self.draws, self.tree.num_periods))
        return disaster

    def _disaster_cons_simulation(self):
        """Simulates consumption conditional on disaster, based on the parameter disaster_tail.

        Args:
            draws (int): Number of draws from distribution.
    
        Returns:
            ndarray

        """
        disaster_cons = self._gamma_array(1.0, self.disaster_tail, self.draws)
        return disaster_cons

    def _interpolation_of_temp(self, temperature): # what to call this one?
        """


        """
        return temperature[:, np.newaxis] * 2.0 * (1.0 - 0.5**(self.tree.decision_times[1:] / self.maxh))
      

    def _economic_impact_of_temp(self, temperature):
        """Economic impact of temperatures, Pindyck [2009].

        Args:
            temperature (ndarray): Array of simulated temperatures.

        Returns:
            ndarray: Consumptions.

        """
        impact = self._pindyck_impact_simulation()
        term1 = -2.0 * impact[:, np.newaxis] * self.maxh * temperature[:,np.newaxis] / np.log(0.5)
        term2 = (self.cons_growth - 2.0 * impact[:, np.newaxis] \
                * temperature[:, np.newaxis]) * self.tree.decision_times[1:]
        term3 = (2.0 * impact[:, np.newaxis] * self.maxh \
                * temperature[:, np.newaxis] * 0.5**(self.tree.decision_times[1:] / self.maxh)) / np.log(0.5)
        return np.exp(term1 + term2 + term3)

    def _tipping_point_update(self, tmp, consump, peak_temp_interval=30.0):
        """Determine whether a tipping point has occurred, if so reduce consumption for 
        all periods after this date.

        Args:
            tmp (ndarray): Array of temperatures.
            damage (ndarray): Array of damages.
            consump (ndarray): Array of consumption.
            peak_temp_interval (float): The normalization of the peak_temp parameter period length, 
                which specifies the probability of a tipping point(temperature) over a given time interval.

        Returns:
            ndarray: Updated consumptions.

        """
        draws = tmp.shape[0]
        disaster = self._disaster_simulation()
        disaster_cons = self._disaster_cons_simulation()
        period_lengths = self.tree.decision_times[1:] - self.tree.decision_times[:-1]
        
        tmp_scale = np.maximum(self.peak_temp, tmp)
        ave_prob_of_survival = 1.0 - np.square(tmp / tmp_scale) 
        prob_of_survival = ave_prob_of_survival**(period_lengths / peak_temp_interval)
        # this part may be done better, this takes a long time to loop over
        res = prob_of_survival < disaster
        rows, cols = np.nonzero(res)
        row, count = np.unique(rows, return_counts=True)
        first_occurance = zip(row, cols[np.insert(count.cumsum()[:-1],0,0)])
        for pos in first_occurance:
            consump[pos[0], pos[1]:] *= np.exp(-disaster_cons[pos[0]])
        return consump

    def _run(self, temperature):
        """Calculate the distribution of damage for specific path. 

        Args:
            temperature (ndarray): Array of simulated temperatures.

        Returns:
            ndarray: Distribution of damages.

        """
        # implementation of the temperature and economic impacts from Pindyck [2012] page 6
        d = np.zeros((self.tree.num_final_states, self.tree.num_periods))
        tmp = self._interpolation_of_temp(temperature)
        consump = self._economic_impact_of_temp(temperature)
        peak_cons = np.exp(self.cons_growth*self.tree.decision_times[1:])
            
        # adding tipping points
        if self.tip_on:
            consump = self._tipping_point_update(tmp, consump)
                
        # sort based on outcome of simulation
        consump = self._sort_array(consump)
        damage = 1.0 - (consump / peak_cons)
        weights = self.tree.final_states_prob*(self.draws)
        weights = (weights.cumsum()).astype(int)
    
        d[0,] = damage[:weights[0], :].mean(axis=0)
        for n in range(1, self.tree.num_final_states):
            d[n,] = np.maximum(0.0, damage[weights[n-1]:weights[n], :].mean(axis=0))
        
        return d
       
    def simulate(self, ret=True, write_to_file=True):
        """Create damage function values in 'p-period' version of the Summers - Zeckhauser model.

        The damage function simulation is a key input into the pricing engine. Damages are 
        represented in arrays of dimension n x p, where n = num states and p = num periods.
        The arrays are created by Monte Carlo simulation. Each array specifies for each state 
        and time period a damage coefficient. GHG levels are increasing along a path that leads
        to a given level at a given time in the future, e.g. a path in which GHG = 1000 ppm in 2200
        a state, 1 ... num_states is an index of the worst to best damage outcomes. For each state
        d(1,i) gives the average percentage damage that occurs at the end of period one in that state.
        For each state d(2,i) gives the average percentage damage that occurs at the end of period two
        in that state, and so on. These d arrays determine the damage in optimizations that follow. 

        Up to a point, the Monte Carlo follows Pindyck (2012) Uncertain Outcomes and Climate Change 
        Policy:
            - There is a gamma distribution for temperature
            - There is a gamma distribution for economic impact (conditional on temp)

        However, in addition, this program adds a probability of a tipping point (conditional on temp).
        This probability is a decreasing function of the parameter peak_temp, conditional on a tipping
        point. Damage itself is a decreasing function of the parameter disaster_tail.

        Args:
            dist_params (ndarray or list): Distribution parameters for temperature simulation.
            draws (int): Number of samples drawn in Monte Carlo simulation.

        Returns:
            ndarray: 3D-array of simulated damages.

        """
        dnum = len(self.ghg_levels)
        self.peak_cons = np.exp(self.cons_growth*self.tree.decision_times[1:])

        if self.temp_map == 0:
            temperature = self._pindyck_simulation()
        elif self.temp_map == 1:
            temperature = self._ww_simulation()
        elif self.temp_map == 2:
            temperature = self._rb_simulation()
        elif self.temp_map == 3:
            temperature = self._normal_simulation()
        elif self.temp_map == 4:
            temperature = self._gamma_simulation()
        else:
            raise ValueError("temp_map not in interval 0-4")

        pool = mp.Pool(processes=dnum)
        self.d = np.array(pool.map(self._run, temperature))

        if write_to_file:
            self._write_to_file()
        if ret:
            return self.d


    
# save

"""   
self.tree = tree
self.peak_temp = peak_temp
self.disaster_tail = disaster_tail
self.tip_on = tip_on
self.temp_map = temp_map
self.pindyck_impact_k = pindyck_impact_k
self.pindyck_impact_theta = pindyck_impact_theta
self.pindyck_impact_displace = pindyck_impact_displace
self.temp_map = temp_map
self.dist_params = dist_params
self.maxh = maxh
self.cons_growth = cons_growth
self.ghg_levels = ghg_levels
self.d = None

# self, my_tree, ghg_levels, peak_temp=9.0, disaster_tail=13.0, tip_on=True, 
#    temp_map=1, dist_params=None, pindyck_impact_k=4.5, 
#    pindyck_impact_theta=21341.0, pindyck_impact_displace=-0.0000746, maxh=100.0,
#    cons_growth=0.02):
"""















