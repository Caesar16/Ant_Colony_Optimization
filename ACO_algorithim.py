import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math


class AntColonyOptimization:
    def __init__(self, ants, evaporation_rate, intensification, size_raw, size_col, list_of_var, low=1.0, high=10.0, alpha=1.0, beta=0.0, beta_evaporation_rate=0, choose_best=.1):
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
        self.pheromone_intensification = intensification
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
        self.visited_nodes_matrix = []
        self.main_array = []
        self.best_result = []
        #self.map = None
        #self.set_of_available_nodes = None

        # Internal stats
        #self.best_series = []
        #self.best = None
        #self.fitted = False
        #self.best_path = None
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

    def initialize(self):
        for l in range(len(self.list_of_var)):
            self.main_array.append(np.ones((self.size_raw, self.size_col)))
            self.heuristic_matrix.append(np.ones((self.size_raw, self.size_col)))
            self.probability_matrix.append(np.ones((self.size_raw, self.size_col)))
            self.pheromone_matrix.append(np.ones((self.size_raw, self.size_col)))
            for i in range(self.size_raw):
                for j in range(self.size_col):
                    self.main_array[l][i][j] = abs(round(random.uniform(self.low, self.high), 2))
                    decimal, whole = math.modf(self.main_array[l][i][j])
                    if decimal == 0:
                        self.main_array[l][i][j] = self.main_array[l][i][j] + .1
                    else:
                        assert decimal != 0, "Can't be a zero!"

    def _evaporation(self):
        """
        Evaporate some pheromone as the inverse of the evaporation rate.  Also evaporates beta if desired.
        """
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        self.heuristic_beta *= (1 - self.beta_evaporation_rate)

    def remove_node(self, node):
        self.main_array.remove(node)

    def computation(self, indx, raw, column, a):
        self.heuristic_matrix[indx][raw][column] = (a / (abs(a - self.main_array[indx][raw][column])))
        self.pheromone_matrix[indx][raw][column] = ((1 - self.evaporation_rate) * self.pheromone_matrix[indx][raw][column]) + self.pheromone_intensification
        self.probability_matrix[indx][raw][column] = math.pow(self.pheromone_matrix[indx][raw][column], self.heuristic_alpha) * math.pow(self.heuristic_matrix[indx][raw][column], self.heuristic_beta)

    def first_step(self, iterations):
        """
        The goal of this function is find (in random way) first number in the first ever iteration, after that when algorithim pick two numbers will upload the matrices and create new one with new numbers in the center of mastrix.
        :param main_array:
        :param heuristic_matrix:
        :param probability_matrix:
        :param pheromone_matrix:
        :return:
        """
        for iteration in range(iterations):                         # number of initial iteration
            for indx, var in enumerate(self.list_of_var):                 # in this case, 2 variable
                for agent in range(1, self.ants + 1):       # 5 ants
                    for num_col in range(self.size_col):    # column size is 20
                        #if iteration == 0:
                         #   rand_raw = random.randint(0, self.size_raw - 1)
                        #else:
                         #   rand_raw = self.main_array[var][:, num_col].tolist().index(max(self.main_array[var][:, num_col]))
                        rand_raw = random.randint(0, self.size_raw - 1)
                        self.computation(indx, rand_raw, num_col, var)
                #ind = np.unravel_index(np.argmax(np.amax(self.heuristic_matrix))
                #self.best_result.append([index_min])



    def evaluate(self, main_array, heuristic_matrix, probability_matrix, pheromone_matrix):
        for agent in range(1, self.ants + 1):
            for num_col in self.size_col:
                choose = main_array[random.randint(0, self.size_raw - 1), num_col]

    def hello(self):
        self.initialize()
        self.first_step(100)
        print(self.heuristic_matrix)
        print(self.main_array)

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
d = AntColonyOptimization(5, 3, 2, 20, 20, [3,4])
y = d.hello()





