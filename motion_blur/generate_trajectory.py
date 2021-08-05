import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# most of the code lifted from: http://home.deib.polimi.it/boracchi/Projects/PSFGeneration.html

class Trajectory(object):
    def __init__(self, canvas=64, iters=2000, max_len=60, expl=None):
        """
        Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012]. Each
        trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
        2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration,
        is affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the
        previous particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming
        at inverting the particle velocity may arises, mimicking a sudden movement that occurs when the user presses
        the camera button or tries to compensate the camera shake. At each step, the velocity is normalized to
        guarantee that trajectories corresponding to equal exposures have the same length. Each perturbation (
        Gaussian, inertial, and impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi
        2011] can be obtained by setting anxiety to 0 (when no impulsive changes occurs
        :param canvas: size of domain where our trajectory os defined.
        :param iters: number of iterations for definition of our trajectory.
        :param max_len: maximum length of our trajectory.
        :param expl: this param helps to define probability of big shake. Recommended expl = 0.005.
        """
        self.canvas = canvas
        self.iters = iters
        self.max_len = max_len
        if expl is None:
            self.expl = 0.1 * np.random.uniform(0, 1)
        else:
            self.expl = expl

        self.tot_length = None
        self.big_expl_count = None
        self.x = None
        self.unprocessedX = None

    def fit(self):
        """
        Generate motion, you can save or plot, coordinates of motion you can find in x property.
        Also you can find properties tot_length, big_expl_count.
        :return: x (vector of motion).
        """
        tot_length = 0
        big_expl_count = 0
        # how to be near the previous position

        centripetal = 0.7 * np.random.uniform(0, 1)
        # probability of big shake
        prob_big_shake = 0.2 * np.random.uniform(0, 1)
        # term determining, at each sample, the random component of the new direction
        gaussian_shake = 10 * np.random.uniform(0, 1)
        init_angle = 360 * np.random.uniform(0, 1)

        img_v0 = np.sin(np.deg2rad(init_angle))
        real_v0 = np.cos(np.deg2rad(init_angle))

        v0 = complex(real=real_v0, imag=img_v0)
        v = v0 * self.max_len / (self.iters - 1)

        if self.expl > 0:
            v = v0 * self.expl

        
        # start off with zero at the center.
        x = np.array([complex(real=0, imag=0)] * (self.iters))

        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count += 1
            else:
                next_direction = 0

            dv = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
                                      self.max_len / (self.iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (self.max_len / float((self.iters - 1)))
            x[t + 1] = x[t] + v
            tot_length = tot_length + abs(x[t + 1] - x[t])

        # centering the startpoint

        self.unprocessedX = np.copy(x)

        # make sure it fits within the boundary but also ensure the start point is at the center of the canvas.
        # x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
        # x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
        # x += complex(real=ceil((self.canvas - max(x.real)) / 2), imag=ceil((self.canvas - max(x.imag)) / 2))
        x = x + complex(self.canvas/2, self.canvas/2)

        self.tot_length = tot_length
        self.big_expl_count = big_expl_count
        self.x = x

        return self

    def applyscale_factor(self):
        x = self.unprocessedX
        scalingFactor = np.max([np.max(-1* x.real/((self.canvas/2)-2)), np.max(-1* x.imag/((self.canvas/2)-2)), np.max(x.real/((self.canvas/2)-2)), np.max(x.imag/((self.canvas/2)-2))])
        self.x = x / scalingFactor
        self.x = self.x + complex(self.canvas/2, self.canvas/2)