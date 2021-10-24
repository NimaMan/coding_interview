import numpy as np

import matplotlib.pyplot as plt


class RandomWalk1D():
    def __init__(self, ):
        self.p = 1/2
        self.max_number_steps = int(1e8)
        self.init_state = 0  # (x)
        self.states_trajectory = [self.init_state]
        self.absorbing_states = [2]
    
    def one_step_transition(self, current_state):
        rnd = np.random.random()
        if rnd < self.p:
            new_state = current_state + 1
        else:
            new_state = current_state - 1

        return new_state

    def get_num_steps_to_absorbing_states_in_set(self, track_states=False,
                                                 init_state=0):
        num_steps = 0
        current_state = init_state
        while current_state not in self.absorbing_states:
            current_state = self.one_step_transition(current_state)
            if track_states:
                self.states_trajectory.append(current_state)
            if num_steps > self.max_number_steps:
                return num_steps
            num_steps += 1
            
        return num_steps
               
    def run_experiments_parallel(self, experiment_func, num_samples, 
                                 mp_num_processors=4):
        import multiprocessing as mp 
        pool = mp.Pool(processes=mp_num_processors)
        num_steps_samples = [
            pool.apply_async(experiment_func)
            for idx in range(num_samples)
            ]
        
        num_steps_samples = [result.get() for result in num_steps_samples]
            
        return num_steps_samples
    
    def run_experiments_serial(self, experiment_func, num_samples):
        num_steps_samples = []
        for _ in range(num_samples):
            num_steps = experiment_func()
            num_steps_samples.append(num_steps)
        
        return num_steps_samples
    

class RandomWalk2D():
    def __init__(self, ):
        self.p = 1/4
        self.max_number_steps = int(1e8)
        self.init_state = (0, 0)  # (x, y)
        self.states_trajectory = [self.init_state]
        self.absorbing_states = {(2, 2), (2, -2), (-2, 2), (-2, -2)}
    
    @staticmethod
    def puzzle_1_boundry_func(state):
        
        return ((abs(state[0]) < 2) and (abs(state[1]) < 2))
    
    @staticmethod
    def puzzle_3_boundry_func(state):
        
        return ((state[0]-0.25)/3)**2 + ((state[1]-0.25)/4)**2 < 1
    
    def one_step_transition(self, current_state):
        rnd = np.random.random()
        new_state = [None, None]
        if rnd < self.p:
            new_state[0] = current_state[0] + 1
            new_state[1] = current_state[1]
        elif (rnd >= self.p) & (rnd < 2*self.p):
            new_state[0] = current_state[0] - 1
            new_state[1] = current_state[1]
        elif (rnd >= 2*self.p) & (rnd < 3 * self.p):
            new_state[0] = current_state[0]
            new_state[1] = current_state[1] + 1
        else:
            new_state[0] = current_state[0]
            new_state[1] = current_state[1] - 1

        return tuple(new_state)

    def get_num_steps_to_absorbing_states_in_set(self, track_states=False,
                                                 init_state=(0, 0)):
        num_steps = 0
        current_state = init_state
        while current_state not in self.absorbing_states:
            current_state = self.one_step_transition(current_state)
            if track_states:
                self.states_trajectory.append(current_state)
            if num_steps > self.max_number_steps:
                return num_steps
            num_steps += 1
            
        return num_steps
    
    def get_num_steps_to_absorbing_states_on_linear_line(self, p1=(1, 0),
                                                         p2=(0, 1), 
                                                         track_states=False,
                                                         init_state=(0, 0)):
        line_slope = (p1[1]- p2[1])/(p1[0]- p2[0])
        line_intercept = p1[1] - p1[0] * line_slope
        
        num_steps = 0
        current_state = init_state
        while (current_state[1] - current_state[0]*line_slope) != line_intercept:
            current_state = self.one_step_transition(current_state)
            if track_states:
                self.states_trajectory.append(current_state)
            if num_steps > self.max_number_steps:
                return num_steps
            num_steps += 1
        return num_steps     

    def get_num_steps_to_absorbing_states_outside_boundry(self, 
                                                           track_states=False,
                                                           init_state=(0, 0)):
        
        num_steps = 0
        current_state = init_state
        while self.puzzle_1_boundry_func(current_state):
            current_state = self.one_step_transition(current_state)
            if track_states:
                self.states_trajectory.append(current_state)
            if num_steps > self.max_number_steps:
                return num_steps
            num_steps += 1
            
        return num_steps
    
    def get_neighboring_states(self, current_state):
        i, j = current_state
        
        return ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
        
    def get_states_inside_baoundry(self, boundry_func):
        current_state = (0, 0)
        is_explored = []
        to_explore = [current_state]
        states_in_boundry = {(0, 0)}
        while True:
            new_states = self.get_neighboring_states(current_state)
            for state in new_states:
                if boundry_func(state):
                    states_in_boundry.add(state)
                    if state not in is_explored:
                        to_explore.append(state)
                        
            to_explore.remove(current_state)
            is_explored.append(current_state)
            if len(to_explore)>=1:
                current_state = to_explore[-1]
            else:
                break 
            
        return states_in_boundry
    
    def get_states_in_boundry_transition_matrix(self, boundry_func):
        states_in_boundry = list(self.get_states_inside_baoundry(boundry_func))
        
        probability_transition = np.zeros((len(states_in_boundry), 
                                           len(states_in_boundry)))
        
        for state_idx, state in enumerate(states_in_boundry):
            neighbor_states = self.get_neighboring_states(state)
            p = 0
            for ns in neighbor_states:
                if ns in states_in_boundry:
                    ns_idx = states_in_boundry.index(ns)
                    probability_transition[state_idx, ns_idx] = 0.25
                    
        return probability_transition, states_in_boundry
    
    def get_mean_time_to_boundry_func_analytic(self, boundry_func):
        p, s = self.get_states_in_boundry_transition_matrix(boundry_func)
        n = p.shape[0]
        origin_idx = s.index((0, 0))
        return np.linalg.inv(np.eye(n) - p)[origin_idx].sum()
        
    
    def run_experiments_parallel(self, experiment_func, num_samples, 
                                 mp_num_processors=4):
        import multiprocessing as mp 
        pool = mp.Pool(processes=mp_num_processors)
        num_steps_samples = [
            pool.apply_async(experiment_func)
            for idx in range(num_samples)
            ]
        
        num_steps_samples = [result.get() for result in num_steps_samples]
            
        return num_steps_samples
    
    def run_experiments_serial(self, experiment_func, num_samples):
        num_steps_samples = []
        for _ in range(num_samples):
            num_steps = experiment_func()
            num_steps_samples.append(num_steps)
        
        return num_steps_samples
    


if __name__ == "__main__":
    
    
    import pickle
    rw = RandomWalk2D()
    num_samples = int(1e6)
    experiment_func = rw.get_num_steps_to_absorbing_states_outside_boundry

    samples = rw.run_experiments_parallel(experiment_func, num_samples, 
                                          mp_num_processors=10)
    #mean_number_of_steps = np.mean(samples)
    #res = rw.run_experiments_parallel(experiment_func, num_samples)
    
    #with open("num_samples_absorbing_boundry_func.pkl", "wb") as f:
    #    pickle.dump(res, f)
           
        
    
    
    
    
    
    
    
    
    