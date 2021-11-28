import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math


class AntColonyOptimization:
    def __init__(self, ants, evaporation_rate, pheromone_intensification, size_raw, size_col, list_of_var, low=1.0, high=10.0, alpha=1.0, beta=0.0, beta_evaporation_rate=0, choose_best=.1):
        """
               Ant colony optimizer.  Traverses a graph and finds either the max or min distance between nodes.
               :param ants: number of ants to traverse the graph
               :param evaporation_rate: rate at which pheromone evaporates
               :param intensification: constant added to the best path
               :param alpha: weighting of pheromone
               :param beta: weighting of heuristic (1/distance)
               :param beta_evaporation_rate: rate at which beta decays (optional)
               :param choose_best: probability to choose the best route
        """
        # Parameters
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_intensification = pheromone_intensification
        self.heuristic_alpha = alpha
        self.heuristic_beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.choose_best = choose_best
        self.size_raw = size_raw
        self.size_col = size_col
        self.low = low
        self.high = high
        self.list_of_var = list_of_var

        # Internal representations
        self.pheromone_matrix = []
        self.heuristic_matrix = []
        self.probability_matrix = []
        self.cost_function = []
        self.main_array = []
        self.best_result = []
        self.x1 = None
        self.y1 = []
        self.a = None
        self.b = None
        self.n = None
        #self.map = None
        #self.set_of_available_nodes = None

        # Internal stats
        #self.best_series = []
        #self.best = None
        #self.fitted = False
        self.best_path_per_iter = None
        self.fit_time = None

        # Plotting values
        self.stopped_early = False
        print("hello ant")

    def __str__(self):
        string = "Ant Colony Optimizer"
        string += "\n--------------------"
        string += "\nDesigned to optimize either the minimum or maximum distance between nodes in a square matrix that behaves like a distance matrix."
        string += "\n--------------------"
        string += f"\nNumber of ants:\t\t\t\t{self.ants}"
        string += f"\nEvaporation rate:\t\t\t{self.evaporation_rate}"
        string += f"\nIntensification factor:\t\t{self.pheromone_intensification}"
        string += f"\nAlpha Heuristic:\t\t\t{self.heuristic_alpha}"
        string += f"\nBeta Heuristic:\t\t\t\t{self.heuristic_beta}"
        string += f"\nBeta Evaporation Rate:\t\t{self.beta_evaporation_rate}"
        string += f"\nChoose Best Percentage:\t\t{self.choose_best}"
        string += "\n--------------------"
        string += "\nUSAGE:"
        string += "\nNumber of ants influences how many paths are explored each iteration."
        string += "\nThe alpha and beta heuristics affect how much influence the pheromones or the distance heuristic weigh an ants' decisions."
        string += "\nBeta evaporation reduces the influence of the heuristic over time."
        string += "\nChoose best is a percentage of how often an ant will choose the best route over probabilistically choosing a route based on pheromones."
        string += "\n--------------------"
        return string

    def sinus_function(self, a=3, b=4, n=1000):
        self.a = a
        self.b = b
        self.n = n
        self.x1 = np.linspace(0, 2*math.pi, self.n, endpoint=True)
        self.y1 = [self.a*math.sin(self.b*x) for x in self.x1]
        plt.plot(self.x1, self.y1, 'o')
        plt.ylim([-4, 4])
        plt.show()

    def initialize(self):
        self.main_array = np.empty((self.size_raw, self.size_col), dtype=object)
        self.heuristic_matrix = np.empty((self.size_raw, self.size_col), dtype=object)
        self.probability_matrix = np.empty((self.size_raw, self.size_col), dtype=object)
        self.pheromone_matrix = np.empty((self.size_raw, self.size_col), dtype=object)
        self.cost_function = np.empty((self.size_raw, self.size_col), dtype=object)
        for i in range(self.size_raw):
            for j in range(self.size_col):
                self.main_array[i][j] = [round(random.uniform(self.low, self.high), 2), round(random.uniform(self.low, self.high), 2)]

    def _evaporation(self):
        """
        Evaporate some pheromone as the inverse of the evaporation rate.  Also evaporates beta if desired.
        """
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        self.heuristic_beta *= (1 - self.beta_evaporation_rate)

    def computation(self, raw, column):
        diff_1 = 0
        diff_2 = 0
        #self.heuristic_matrix[raw][column] = (a / (abs(a - self.main_array[raw][column])))
        for each in range(self.n):
            diff_1 += abs(self.y1[each] - (self.main_array[raw][column][0]*math.sin(self.main_array[raw][column][1]*self.x1[each])))
            diff_2 += abs(self.y1[each] - (self.main_array[raw][column][1]*math.sin(self.main_array[raw][column][0]*self.x1[each])))
        if diff_1 >= diff_2:
            best_diff = diff_2
        else:
            best_diff = diff_1
        self.cost_function[raw][column] = best_diff / self.n

        #self.pheromone_matrix[raw][column] = ((1 - self.evaporation_rate) * self.pheromone_matrix[raw][column]) + self.pheromone_intensification
        #self.probability_matrix[raw][column] = math.pow(self.pheromone_matrix[raw][column], self.heuristic_alpha) * math.pow(self.cost_function[raw][column], self.heuristic_beta)

    def epoch(self, procent):
        best_epoch_value = np.amin(self.cost_function)
        self.main_array = np.empty((self.size_raw, self.size_col), dtype=object)
        prc_per_node = procent / (self.size_raw * self.size_col)
        start = best_epoch_value - (best_epoch_value * (procent / 2))
        for x_ in range(self.size_col):
            for y_ in range(self.size_raw):
                self.main_array[x_][y_] = start * prc_per_node

    def fitted(self, iterations, epoch_num = 100, procent = .45, procent_dec = .4):
        """
        The goal of this function is find (in random way) first number in the first ever iteration, after that when algorithim pick two numbers will upload the matrices and create new one with new numbers in the center of mastrix.
        :param main_array:
        :param heuristic_matrix:
        :param probability_matrix:
        :param pheromone_matrix:
        :return:
        """

        for iteration in range(1, iterations + 1):                         # number of initial iteration
            for agent in range(1, self.ants + 1):       # 5 ants
                for num_col in range(self.size_col):    # column size is 5
                    #if iteration == 0:
                    #   rand_raw = random.randint(0, self.size_raw - 1)
                    #else:
                    #   rand_raw = self.main_array[var][:, num_col].tolist().index(max(self.main_array[var][:, num_col]))
                    rand_raw = random.randint(0, self.size_raw - 1)
                    self.computation(rand_raw, num_col)
                    if iter == epoch_num:
                        self.epoch(procent - procent_dec)
                        self.fitted(iterations - iteration, epoch_num, procent - procent_dec)
        #best = np.amin(self.cost_function)
        #matrix = np.array(self.main_array)
        #best_ind = np.where(matrix == best)
        best_ind_raw = np.argmin(self.cost_function, axis=1)
        best_ind_col = np.argmin(self.cost_function, axis=0)
        print(self.main_array[best_ind_raw, best_ind_col])



    def plot(self):
        """
        Plots the score over time after the model has been fitted.
        :return: plot
        """

        fig, ax = plt.subplots(figsize=(20, 15))
        ax.plot(self.best_series, label="Best Run")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Performance")
        ax.text(.8, .6,
                'Ants: {}\nEvaporation: {}\nIntensify: {}\nAlpha: {}\nBeta: {}\nBeta Evap: {}\nChoose Best: {}\n\nFit Time: {}m{}'.format(
                    self.ants, self.evaporation_rate, self.pheromone_intensification, self.heuristic_alpha,
                    self.heuristic_beta, self.beta_evaporation_rate, self.choose_best, self.fit_time // 60,
                    ["\nStopped Early!" if self.stopped_early else ""][0]),
                bbox={'facecolor': 'gray', 'alpha': 0.8, 'pad': 10}, transform=ax.transAxes)
        ax.legend()
        plt.title("Ant Colony Optimization Results (best: {})".format(np.round(self.best, 2)))
        plt.show()



random.seed( 30 )
d = AntColonyOptimization(5, 3, 2, 9, 9, [3,4])
d.initialize()
d.sinus_function()
d.fitted(1000)




