import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from motion_blur.generate_trajectory import Trajectory
import time
from PIL import Image
from matplotlib import cm

class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None, path_to_save=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory_obj = Trajectory(canvas=canvas, expl=0.005)
            self.trajectory_obj.fit(show=True, save=False)
            self.trajectory = self.trajectory_obj.x
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1/100, 1/10, 1/2, 1]
        else:
            self.fraction = fraction
            
        self.path_to_save = path_to_save
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self, show=False, save=False):
        PSF = np.zeros(self.canvas)

        start_time = time.perf_counter()

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                #looks like checks to see where to stop according to the fraction provided.
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                #some boundary checks then some interpolation
                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )

            self.PSFs.append(PSF / (self.iters))

        #print(round((time.perf_counter() - start_time)*1000, 2))
        if show or save:
            self.plot_canvas(show, save)

        return self.PSFs

    def plot_canvas(self, show, save):
        if len(self.PSFs) == 0:
            raise Exception("Please run fit() method first.")
        else:
            if show:
                plt.close()
                fig, axes = plt.subplots(1, self.PSFnumber, figsize=(10, 10))
                for i in range(self.PSFnumber):
                    axes[i].imshow(self.PSFs[i], cmap='gray')

            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                psfImage = np.uint8(255*(self.PSFs[0]/np.max(self.PSFs[0])))
                Image.fromarray(psfImage).save("imagenetDebugPhotos/psf.png")
            elif show:
                plt.show()

    def centerPSF(self):
        psf = self.PSFs[0]
        totalSum = np.sum(self.PSFs[0])

        non_zero_indices = np.nonzero(psf > 0)

        averageSum = [0,0]
        for coordX, coordY in zip(non_zero_indices[1], non_zero_indices[0]):
            weight = self.PSFs[0][coordY, coordX]/totalSum

            averageSum[0] += coordX * weight
            averageSum[1] += coordY * weight

        offsetX = int(averageSum[0] - self.canvas[0]/2)
        offsetY = int(averageSum[1] - self.canvas[1]/2)

        self.PSFs[0] = np.roll(self.PSFs[0], shift = -offsetX, axis = 1)
        self.PSFs[0] = np.roll(self.PSFs[0], shift = -offsetY, axis = 0)

    def findOffsets(self):
        leftExpansion = 0
        rightExpansion = 0

        bottomExpansion = 0
        topExpansion = 0
        psf = self.PSFs[0]
        nonZeroInd = np.nonzero(psf > 0)


        for coordX, coordY in zip(nonZeroInd[1], nonZeroInd[0]):
            offsetX = coordX-(self.canvas[0]/2 - 1)
            if offsetX > 0 and offsetX > rightExpansion:
                rightExpansion = offsetX
            elif offsetX <= 0 and -offsetX > leftExpansion:
                leftExpansion = -offsetX

            offsetY = coordY-(self.canvas[1]/2 - 1)
            if offsetY > 0 and offsetY > bottomExpansion:
                bottomExpansion = offsetY
            elif offsetY <= 0 and -offsetY > topExpansion:
                topExpansion = -offsetY
        
        return [leftExpansion, topExpansion, rightExpansion, bottomExpansion]

if __name__ == '__main__':
    psf = PSF(canvas=128, path_to_save='/Users/mykolam/PycharmProjects/University/RandomMotionBlur/psf.png')
    psf.fit(show=True, save=True)