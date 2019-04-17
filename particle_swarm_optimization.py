import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.evaluation = -inf

        self.x = np.random.uniform(lower_bound, upper_bound)
        self.best_position = self.x
        self.best_evaluation = -inf
        delta = upper_bound - lower_bound
        self.v = np.random.uniform(-delta, delta)


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.hyperparams = hyperparams
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.particles = [Particle(lower_bound, upper_bound) for _ in range(hyperparams.num_particles)]
        self.idx = 0  # index of particle simulation order

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        return sorted(self.particles, key=lambda x: x.evaluation)[-1].x

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        return sorted(self.particles, key=lambda x: x.evaluation)[-1].evaluation

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        return self.particles[self.idx].x

    def advance_generation(self):
        """
        Advances the generation of particles.
        """
        w = self.hyperparams.inertia_weight
        phi_p = self.hyperparams.cognitive_parameter
        phi_g = self.hyperparams.social_parameter

        for particle in self.particles:
            rp = random.uniform(0, 1)
            rg = random.uniform(0, 1)
            particle.v = (w * particle.v +
                          phi_p * rp * (particle.best_position - particle.x) +
                          phi_g * rg * (self.get_best_position() - particle.x))
            particle.x = particle.x + particle.v

            if particle.evaluation > particle.best_evaluation:
                particle.best_position = particle.x

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # register evaluation
        part = self.particles[self.idx]
        part.evaluation = value

        # increment index
        self.idx += 1
        if self.idx == len(self.particles):
            self.advance_generation()
            self.idx = 0

