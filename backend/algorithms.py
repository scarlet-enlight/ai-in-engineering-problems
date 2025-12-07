"""
Implementations of optimization algorithms
"""

import numpy as np
from base_algorithm import BaseOptimizer


class PSO(BaseOptimizer):
    """Particle Swarm Optimization"""
    
    def __init__(self, *args, w=0.729, c1=1.49445, c2=1.49445, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = w  # Inertia weight (standard: 0.729)
        self.c1 = c1  # Cognitive coefficient (standard: 1.49445)
        self.c2 = c2  # Social coefficient (standard: 1.49445)
        self.V = np.zeros((self.N, self.D))
        self.pbest = None
        self.pbest_fitness = None
    
    def initialize_population(self):
        super().initialize_population()
        # Initialize velocities to small random values
        v_max = (self.ub - self.lb) * 0.2
        self.V = self.rng.uniform(-v_max, v_max, (self.N, self.D))
        self.pbest = self.X.copy()
        self.pbest_fitness = self.f.copy()
    
    def iterate(self):
        # Update all particles
        for i in range(self.N):
            r1, r2 = self.rng.random(2)
            
            # Update velocity
            self.V[i] = (self.w * self.V[i] + 
                        self.c1 * r1 * (self.pbest[i] - self.X[i]) +
                        self.c2 * r2 * (self.best_position - self.X[i]))
            
            # Velocity clamping
            v_max = (self.ub - self.lb) * 0.2
            self.V[i] = np.clip(self.V[i], -v_max, v_max)
            
            # Update position
            self.X[i] = self.clip_to_bounds(self.X[i] + self.V[i])
            
            # Evaluate
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1
            
            # Update personal best
            if self.f[i] < self.pbest_fitness[i]:
                self.pbest[i] = self.X[i].copy()
                self.pbest_fitness[i] = self.f[i]


class GA(BaseOptimizer):
    """Genetic Algorithm"""
    
    def __init__(self, *args, mutation_rate=0.1, crossover_rate=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def iterate(self):
        # Tournament selection
        new_pop = []
        for _ in range(self.N):
            i1, i2 = self.rng.choice(self.N, 2, replace=False)
            parent = self.X[i1].copy() if self.f[i1] < self.f[i2] else self.X[i2].copy()
            new_pop.append(parent)
        
        # Crossover
        offspring = []
        for i in range(0, self.N, 2):
            p1, p2 = new_pop[i], new_pop[min(i+1, self.N-1)]
            if self.rng.random() < self.crossover_rate:
                alpha = self.rng.random()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1.copy(), p2.copy()])
        
        # Mutation
        for i in range(self.N):
            if self.rng.random() < self.mutation_rate:
                j = self.rng.integers(0, self.D)
                offspring[i][j] = self.lb[j] + self.rng.random() * (self.ub[j] - self.lb[j])
            self.X[i] = self.clip_to_bounds(offspring[i])
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1


class GWO(BaseOptimizer):
    """Grey Wolf Optimizer"""
    
    def iterate(self):
        a = 2 - 2 * self.current_iter / self.max_iter
        
        # Sort and get alpha, beta, delta (best three wolves)
        order = np.argsort(self.f)
        alpha = self.X[order[0]].copy()
        beta = self.X[order[1]].copy() if len(order) > 1 else alpha.copy()
        delta = self.X[order[2]].copy() if len(order) > 2 else alpha.copy()
        
        for i in range(self.N):
            # Alpha wolf influence
            r1, r2 = self.rng.random(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha - self.X[i])
            X1 = alpha - A1 * D_alpha
            
            # Beta wolf influence
            r1, r2 = self.rng.random(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta - self.X[i])
            X2 = beta - A2 * D_beta
            
            # Delta wolf influence
            r1, r2 = self.rng.random(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta - self.X[i])
            X3 = delta - A3 * D_delta
            
            # Update position (average of three influences)
            self.X[i] = self.clip_to_bounds((X1 + X2 + X3) / 3.0)
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1


class ABC(BaseOptimizer):
    """Artificial Bee Colony"""
    
    def __init__(self, *args, limit=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.trials = np.zeros(self.N)
    
    def iterate(self):
        # Employed bee phase
        for i in range(self.N):
            k = self.rng.choice([j for j in range(self.N) if j != i])
            phi = self.rng.uniform(-1, 1, self.D)
            v = self.X[i] + phi * (self.X[i] - self.X[k])
            v = self.clip_to_bounds(v)
            f_v = self.func(v)
            self.total_evals += 1
            
            if f_v < self.f[i]:
                self.X[i] = v
                self.f[i] = f_v
                self.trials[i] = 0
            else:
                self.trials[i] += 1
        
        # Onlooker bee phase
        fitness_vals = 1 / (1 + self.f)
        probs = fitness_vals / fitness_vals.sum()
        
        for _ in range(self.N):
            i = self.rng.choice(self.N, p=probs)
            k = self.rng.choice([j for j in range(self.N) if j != i])
            phi = self.rng.uniform(-1, 1, self.D)
            v = self.X[i] + phi * (self.X[i] - self.X[k])
            v = self.clip_to_bounds(v)
            f_v = self.func(v)
            self.total_evals += 1
            
            if f_v < self.f[i]:
                self.X[i] = v
                self.f[i] = f_v
                self.trials[i] = 0
        
        # Scout bee phase
        for i in range(self.N):
            if self.trials[i] > self.limit:
                self.X[i] = self.lb + self.rng.random(self.D) * (self.ub - self.lb)
                self.f[i] = self.func(self.X[i])
                self.total_evals += 1
                self.trials[i] = 0


class BOA(BaseOptimizer):
    """Butterfly Optimization Algorithm"""
    
    def __init__(self, *args, c=0.01, a=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.a = a
    
    def iterate(self):
        # Calculate fragrance for each butterfly based on fitness
        # Better fitness = stronger fragrance
        f_min = np.min(self.f)
        f_max = np.max(self.f)
        
        # Normalize fitness and calculate fragrance
        if f_max - f_min > 1e-10:
            normalized_f = (self.f - f_min) / (f_max - f_min + 1e-10)
        else:
            normalized_f = np.zeros_like(self.f)
        
        # Fragrance is inversely proportional to fitness (lower fitness = better = higher fragrance)
        I = self.c * (1.0 - normalized_f) ** self.a
        
        for i in range(self.N):
            r = self.rng.random()
            if r < 0.5:
                # Global search phase
                j = self.rng.integers(0, self.N)
                self.X[i] = self.X[i] + (r ** 2) * self.best_position - self.X[j] * I[i]
            else:
                # Local search phase
                j, k = self.rng.choice(self.N, 2, replace=False)
                self.X[i] = self.X[i] + (r ** 2) * self.X[j] - self.X[k] * I[i]
            
            self.X[i] = self.clip_to_bounds(self.X[i])
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1


class BAT(BaseOptimizer):
    """Bat Algorithm"""
    
    def __init__(self, *args, fmin=0, fmax=2, A=0.5, r=0.5, alpha=0.9, gamma=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.A = np.full(self.N, A)
        self.r = np.full(self.N, r)
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros((self.N, self.D))
    
    def iterate(self):
        for i in range(self.N):
            freq = self.fmin + (self.fmax - self.fmin) * self.rng.random()
            self.V[i] = self.V[i] + (self.X[i] - self.best_position) * freq
            x_new = self.X[i] + self.V[i]
            
            if self.rng.random() > self.r[i]:
                x_new = self.best_position + 0.001 * self.rng.standard_normal(self.D)
            
            x_new = self.clip_to_bounds(x_new)
            f_new = self.func(x_new)
            self.total_evals += 1
            
            if f_new < self.f[i] and self.rng.random() < self.A[i]:
                self.X[i] = x_new
                self.f[i] = f_new
                self.A[i] *= self.alpha
                self.r[i] = self.r[i] * (1 - np.exp(-self.gamma * self.current_iter))


class SSA(BaseOptimizer):
    """Salp Swarm Algorithm"""
    
    def iterate(self):
        c1 = 2 * np.exp(-(4 * self.current_iter / self.max_iter) ** 2)
        
        for i in range(self.N):
            if i == 0:
                # Leader update
                for j in range(self.D):
                    c2, c3 = self.rng.random(2)
                    if c3 < 0.5:
                        self.X[i, j] = self.best_position[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                    else:
                        self.X[i, j] = self.best_position[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
            else:
                # Follower update
                self.X[i] = 0.5 * (self.X[i] + self.X[i-1])
            
            self.X[i] = self.clip_to_bounds(self.X[i])
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1


class WOA(BaseOptimizer):
    """Whale Optimization Algorithm"""
    
    def iterate(self):
        a = 2 - 2 * self.current_iter / self.max_iter
        
        for i in range(self.N):
            r = self.rng.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.rng.uniform(-1, 1)
            p = self.rng.random()
            
            if p < 0.5:
                if np.abs(A) < 1:
                    D = np.abs(C * self.best_position - self.X[i])
                    self.X[i] = self.best_position - A * D
                else:
                    rand_idx = self.rng.integers(0, self.N)
                    X_rand = self.X[rand_idx]
                    D = np.abs(C * X_rand - self.X[i])
                    self.X[i] = X_rand - A * D
            else:
                D = np.abs(self.best_position - self.X[i])
                self.X[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + self.best_position
            
            self.X[i] = self.clip_to_bounds(self.X[i])
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1


class GTO(BaseOptimizer):
    """Gorilla Troops Optimizer"""
    
    def iterate(self):
        beta = 3
        w = 0.8
        
        # Silverback
        silverback_idx = int(np.argmin(self.f))
        
        for i in range(self.N):
            if self.rng.random() < 0.5:
                # Exploration
                r1 = self.rng.random()
                r2 = self.rng.uniform(-1, 1)
                GX = (self.ub - self.lb) * r1 + self.lb
                
                if r2 >= 0:
                    self.X[i] = (self.rng.random() - w) * GX + w * self.X[i]
                else:
                    self.X[i] = GX - self.rng.random() * (GX - self.X[i])
            else:
                # Exploitation
                L = self.rng.uniform(-1, 1, self.D)
                H = self.rng.uniform(-1, 1) * (1 - self.current_iter / self.max_iter)
                
                if np.abs(H) >= 1:
                    self.X[i] = self.X[silverback_idx] - L * np.abs(2 * self.rng.random() * self.X[silverback_idx] - self.X[i])
                else:
                    self.X[i] = self.X[i] - L * (L * self.X[i] - self.X[silverback_idx])
            
            self.X[i] = self.clip_to_bounds(self.X[i])
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1


class AOA(BaseOptimizer):
    """Arithmetic Optimization Algorithm"""
    
    def iterate(self):
        MOA = 1 - (self.current_iter / self.max_iter) ** (1.0 / 3.0)
        MOP = 1 - (self.current_iter / self.max_iter)
        
        for i in range(self.N):
            r1 = self.rng.random()
            
            if r1 > MOA:
                r2 = self.rng.random()
                if r2 < 0.5:
                    self.X[i] = self.best_position * (MOP + self.rng.random() * 0.1)
                else:
                    self.X[i] = self.best_position / (MOP + self.rng.random() * 0.1)
            else:
                r3 = self.rng.random()
                if r3 < 0.5:
                    self.X[i] = self.best_position - MOP * self.rng.random() * (self.ub - self.lb)
                else:
                    self.X[i] = self.best_position + MOP * self.rng.random() * (self.ub - self.lb)
            
            self.X[i] = self.clip_to_bounds(self.X[i])
            self.f[i] = self.func(self.X[i])
            self.total_evals += 1


# Dictionary of available algorithms with full names
ALGORITHMS = {
    'GTOA': None,  # Will be imported from gtoa.py
    'PSO': PSO,
    'GA': GA,
    'GWO': GWO,
    'ABC': ABC,
    'BOA': BOA,
    'BAT': BAT,
    'SSA': SSA,
    'WOA': WOA,
    'GTO': GTO,
    'AOA': AOA
}

# Full names for display in UI
ALGORITHM_NAMES = {
    'GTOA': 'Group Teaching Optimization Algorithm',
    'PSO': 'Particle Swarm Optimization',
    'GA': 'Genetic Algorithm',
    'GWO': 'Grey Wolf Optimizer',
    'ABC': 'Artificial Bee Colony',
    'BOA': 'Butterfly Optimization Algorithm',
    'BAT': 'Bat Algorithm',
    'SSA': 'Salp Swarm Algorithm',
    'WOA': 'Whale Optimization Algorithm',
    'GTO': 'Gorilla Troops Optimizer',
    'AOA': 'Arithmetic Optimization Algorithm'
}