from collections import Callable
from pygene3.gene import FloatGeneMax
from pygene3.organism import MendelOrganism
from pygene3.population import Population
import numpy as np


def gen_minimize(n_pars, fun: Callable, bounds: list=None, args: tuple=None,
                 stop_num=10, atol=1e-10, min_iters=100, max_iters=5000, print_each=False):
    func = fun
    if args is not None:
        if not isinstance(args, tuple):
            args = (args,)
        func = lambda *param: fun(*(param + args))
    if bounds is None:
        bounds = [-100, 100]

    class CvGene(FloatGeneMax):
        """
        Gene which represents the numbers used in our organism
        """
        # genes get randomly generated within this range
        randMin = bounds[0]
        randMax = bounds[1]
    
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
        genome = {'p{}'.format(i): CvGene for i in range(n_pars)}
    
        def fitness(self):
            """
            Implements the 'fitness function' for this species.
            Organisms try to evolve to minimise this function's value
            """
            params = np.array([self['p{}'.format(i)] for i in range(n_pars)])
        
            return func(params)
    
        def __repr__(self):
            return "Converger fitness=%f," % (
                self.fitness()) + ' '.join(
                ['{: .12f}'.format(i) for i in [self['p{}'.format(i)] for i in range(n_pars)]])
    
    pop = Population(species=Converger, init=2, childCount=50, childCull=20)

    params_prev = None
    current_iter = 0
    good_in_row = 0
    params =[]
    while current_iter < max_iters:
        pop.gen()
        best = pop.best()
    
        params = np.array([float(i) for i in str(best).split(',')[1].split()])
    
        if current_iter > min_iters:
            delta_params = abs(params - params_prev)
            if delta_params.max() < atol:
                good_in_row += 1
            else:
                good_in_row = 0
    
        if good_in_row > stop_num:
            break
        
        if print_each:
            print('{0:4d}\t{1:s}'.format(current_iter, '  '.join('{: .6f}'.format(i) for i in params)))
    
        params_prev = params
        current_iter += 1

    return params
