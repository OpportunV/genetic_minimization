from pygene3.gene import FloatGeneMax
from pygene3.organism import MendelOrganism
from pygene3.population import Population
import numpy as np


def gen_minimize(n_pars, func, bounds, stop_num=10, atol=1e-10, min_iters=100, max_iters=5000, args=None, print_each=False):
    N_PARS = n_pars
    FUNC = func
    BOUNDS = bounds
    STOP_NUM_IN_ROW = stop_num
    ATOL = atol
    MIN_N_ITER = min_iters
    MAX_N_ITER = max_iters
    if args is not None:
        if not isinstance(args, tuple):
            args = (args,)
        ARGS = args
        FUNC = lambda *param: func(*(param + ARGS))

    class CvGene(FloatGeneMax):
        """
        Gene which represents the numbers used in our organism
        """
        # genes get randomly generated within this range
        randMin = BOUNDS[0]
        randMax = BOUNDS[1]
    
        # probability of mutation
        mutProb = 0.1
    
        # degree of mutation
        mutAmt = 0.1

    class Converger(MendelOrganism):
        """
        Implements the organism which tries
        to converge a function
        """
        # genome = {'p0': CvGene, 'p1': CvGene}
        genome = {'p{}'.format(i): CvGene for i in range(N_PARS)}
    
        def fitness(self):
            """
            Implements the 'fitness function' for this species.
            Organisms try to evolve to minimise this function's value
            """
            params = np.array([self['p{}'.format(i)] for i in range(N_PARS)])
        
            return FUNC(params)
    
        def __repr__(self):
            return "Converger fitness=%f," % (
                self.fitness()) + ' '.join(
                ['{: .12f}'.format(i) for i in [self['p{}'.format(i)] for i in range(N_PARS)]])
    
    pop = Population(species=Converger, init=2, childCount=50, childCull=20)

    params_prev = None
    current_iter = 0
    NUM_OF_GOOD_IN_ROW = 0
    while current_iter < MAX_N_ITER:
        pop.gen()
        best = pop.best()
    
        params = np.array([float(i) for i in str(best).split(',')[1].split()])
    
        if current_iter > MIN_N_ITER:
            delta_params = abs(params - params_prev)
            if delta_params.max() < ATOL:
                NUM_OF_GOOD_IN_ROW += 1
            else:
                NUM_OF_GOOD_IN_ROW = 0
    
        if NUM_OF_GOOD_IN_ROW > STOP_NUM_IN_ROW:
            break
        
        if print_each:
            print('{0:4d}\t{1:s}'.format(current_iter, '  '.join('{: .6f}'.format(i) for i in params)))
    
        params_prev = params
        current_iter += 1

    return params
