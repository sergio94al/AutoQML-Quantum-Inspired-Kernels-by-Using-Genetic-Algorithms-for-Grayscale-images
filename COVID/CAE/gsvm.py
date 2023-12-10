from deap import base, creator, tools, algorithms
import fitness
import random
import numpy as np
import multiprocessing

def gsvm(nqubits, q_depth, X, y,
         mu=50, lambda_=30, cxpb=0.7, mutpb=0.3, ngen=2000,
         use_pareto=True, verbose=True,processes=11,pool=None,
         debug=True):
    print('multi')
    # if processes:
    #     with multiprocessing.Pool(processes=processes) as pool:       
    bits_puerta = 7
    bits_PCA = 7
    long_cadena = (q_depth * nqubits * bits_puerta)+bits_PCA
    creator.create("FitnessMulti", base.Fitness, weights=[-1.0,1.0])
    creator.create("Individual", list, fitness = creator.FitnessMulti, statistics=dict)
    toolbox = base.Toolbox()
    # pool = multiprocessing.Pool(processes=processes)
    # toolbox.register("map", pool.map)
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("Individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, long_cadena)
    toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.4)
    toolbox.register('select', tools.selNSGA2)
    toolbox.register("evaluate", fitness.Fitness(nqubits, X, y, debug=debug))
    pop = toolbox.Population(n=mu)
    #estadisticas del valor de Acc
    stats1 = tools.Statistics(key=lambda ind: ind.fitness.values[1]) 
    stats1.register('media',np.mean)
    stats1.register('std',np.std)
    stats1.register('max',np.max)
    stats1.register('min',np.min)
    logbook = tools.Logbook()
    pareto = tools.ParetoFront(similar = np.array_equal)
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                             mu, lambda_, cxpb, mutpb, ngen,
                                             stats=stats1,
                                             halloffame=pareto, verbose=verbose)
    pareto.update(pop)
    return pop, pareto, logbook
