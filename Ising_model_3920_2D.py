import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math as math

class IsingModel2D:
    """
    Class for 2D Ising Model
    """
    def __init__(self, N, T):
        """
        Initialising variables for simulation

        N = number of spins along one edge (N**2 total spins)
        T = temperature

        """
        self.N = N
        self.T = T
        self.spins = np.random.choice([-1, 1], size=(N, N))
        self.k = 1
        self.beta = 1 / (self.k * self.T)
        self.epsilon = 1
    
    def get_energy(self, config):
        """
        Calculate the energy of the system in a particular state
        """
        energy = 0
        for i in range(self.N):
            for j in range(self.N):

                interaction = config[i][j]*(
                    config[(i + 1) % self.N][j] +
                    config[i][(j + 1) % self.N]
                )
                energy += interaction

        return -energy / self.N**2
    
    def get_entropy(self, config):
        """
        Method to get the entropy per spin of the system
        returns float
        """
        #calculate spin excess
        spin_excess = 0
        up = 0
        down = 0

        for i in range(0,len(config)):
            for j in range(0,len(config)):
                if config[i][j] == 1:
                    up += 1
                if config[i][j] == -1:
                    down += 1

        s = (up - down)/2

        entropy = self.k*(-(2*s**2)/self.N**2 + 0.5*math.log(2/(math.pi*self.N**2)) + self.N**2*math.log(2))
    
        return entropy / self.N**2
    
    def get_free_energy(self, energy, entropy, i):
        """
        Method to get the free energy of the system
        returns float
        """
        free_energy = energy[i] - self.T*entropy[i]
        return free_energy 
    
    def get_heat_capacity(self, energy):
        """
        Method to get the heat capacity of the system
        return float
        """
        return (np.var(energy))/(self.N**2*self.k*self.T**2)

    def get_magnet(self, config):
        """
        Method to calculate the reduced magnetisation per dipole
        """
        ave_list = []
        for i in range(len(config)):
            ave_list.append(abs(np.average(config[i])))
        return np.average(ave_list)

    def get_spin_excess(self, config):
        #calculate spin excess
        spin_excess = 0
        up = 0
        down = 0

        for i in range(0,len(config)):
            for j in range(0,len(config)):
                if config[i][j] == 1:
                    up += 1
                if config[i][j] == -1:
                    down += 1

        s = (up - down)/2
        return s / self.N**2

    def metropolis(self):
        """
        Method to apply the metropolis algorithm
        """
        for _ in range(0, self.N*self.N):
            i, j = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            old_spin = self.spins[i, j]
            new_spin = -old_spin
            delta_energy = 2 * old_spin * (
                self.spins[(i + 1) % self.N, j] + self.spins[(i - 1) % self.N, j] +
                self.spins[i, (j + 1) % self.N] + self.spins[i, (j - 1) % self.N])

            if delta_energy <= 0 or random.random() < np.exp(
                    -delta_energy / (self.k * self.T)):
                self.spins[i][j] = new_spin

    def simulate(self, num_iterations):
        """
        Method to complete the simulation
        """
        spins_history = [self.spins.copy()]
        energies = []
        free_energies = []
        entropies = []
        magnets = []
        heat_capacities = []
        spin_excesses = []

        #initialise equilibrium
        for i in range(1000):
            self.metropolis()
    
        for i in range(num_iterations):
            #changing the temp over the course of the simulation
            if i < num_iterations/2:
                self.T += 0.01
            elif i > num_iterations/2:
                self.T -= 0.01

            self.metropolis()
            spins_history.append(self.spins.copy())
            energy = self.get_energy(self.spins)
            energies.append(energy)
            entropy = self.get_entropy(self.spins)
            entropies.append(entropy)
            free_energy = self.get_free_energy(energies, entropies, i)
            free_energies.append(free_energy)
            magnet = abs(self.get_magnet(self.spins))
            magnets.append(magnet)
            heat_capacity = self.get_heat_capacity(energies)
            heat_capacities.append(heat_capacity)
            spin_excess = self.get_spin_excess(self.spins)
            spin_excesses.append(spin_excess)

        
        return spins_history, energies, entropies, heat_capacities, free_energies, magnets, spin_excesses

    def visualize_spins(self, spins_history, num_iterations):
        """
        Method to animate the evolution of the system
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame):
            ax.clear()
            ax.imshow(spins_history[frame], cmap='gray', extent=[0, self.N, 0, self.N])
            ax.set_title(f'Iteration: {frame + 1} / {len(spins_history)-1}', fontsize=16)
            ax.set_xlabel(f'T = {self.T}')
            if frame == 0:
                fig.savefig(f'2D initial state T={self.T}.png')
            if frame == num_iterations/4:
                fig.savefig(f'2D Ising q1 point T={self.T}.png')
            if frame == num_iterations/2:
                fig.savefig(f'2D Ising midpoint T={self.T}.png')
            if frame == 3*num_iterations/4:
                fig.savefig(f'2D Ising q2 point T={self.T}.png')
            if frame == num_iterations - 1:
                fig.savefig(f'2D final state T={self.T}.png')

        ani = FuncAnimation(fig, update, frames=num_iterations, interval=50)
        plt.show()

    def visualize_energy(self, energy_history):
        """
        Method to animate the evolution of the energy of the simulation
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(energy_history))

        def update(frame):
            ax.clear()
            ax.plot(x[:frame + 1], energy_history[:frame + 1])
            ax.set_title(f'Iteration: {frame + 1}  -  T = {self.T}', fontsize=16)
            ax.set_xlabel('Iteration', fontsize=14)
            ax.set_ylabel('Energy', fontsize=14)
            


        ani = FuncAnimation(fig, update, frames=len(energy_history), interval=50)
        plt.show()

    def compare_observables(self, energy, entropy, heat_capacity, free_energy, magnet, spin_excess):
        mean_energy = round(np.average(energy),3)
        mean_entropy = round(np.average(entropy),3)
        mean_heat_capcity = round(self.N**2*np.var(energy)/self.T**2,3)
        mean_free_energy = round(np.average(free_energy), 3)
        mean_magnetisation = round(np.average(magnet))
        mean_spin_excess = round(np.average(spin_excess),3)
        delta_energy = round(np.std(energy),3)
        delta_entropy = round(np.std(entropy),3)
        delta_heat_capacity = round(np.std(energy),3)
        delta_free_energy = round(np.std(free_energy),3)
        delta_magnetisation = round(np.std(magnet))

        #print(self.T)
        #print(f'Measured Energy: {mean_energy} +/- {delta_energy}')
        #print(f'Predicted Energy: {round(self.u,3)}')
        #print(f'Measured Entropy: {mean_entropy} +/- {delta_entropy}')
        #print(f'Predicted Entropyy: {round(self.S,3)}')
        #print(f'Measured Heat Capcity: {mean_heat_capcity} +/- {delta_heat_capacity}')
        #print(f'Predicted Heat Capcity: {round(self.c,3)}')
        #print(f'Measured Free Energy: {mean_free_energy} +/- {delta_free_energy}')
        #print(f'Predicted Free Energy: {round(self.f,3)}')
        #print(f'Measured Reduced Magnetisation: {mean_magnetisation} +/- {delta_magnetisation}')

        return mean_energy, delta_energy, mean_entropy, delta_entropy, mean_heat_capcity, delta_heat_capacity, mean_free_energy, delta_free_energy, mean_magnetisation, delta_magnetisation, mean_spin_excess


def plot_observable_t(observable, delta, T, name, N):
    fig, ax= plt.subplots()
    ax.clear()
    ax.set_xlabel(f'Temperature  [$\epsilon / k$]')
    ax.set_title(f'{name} vs Temperature')
    ax.grid()

    if name == 'Energy per dipole':
        ax.set_ylabel(f'{name}  [$\epsilon$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.axvline(2.27, linestyle='dashed', color='grey', label='$T_c$')
    elif name == 'Entropy per dipole':
        ax.set_ylabel(f'{name}  [$k$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.axvline(2.27, linestyle='dashed', color='grey', label='$T_c$')
    elif name == 'Heat Capacity per dipole':
        ax.set_ylabel(f'{name}  [$k$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.axvline(2.27, linestyle='dashed', color='grey')
    elif name == 'Free Energy per dipole':
        ax.set_ylabel(f'{name}  [$\epsilon$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=4, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.axvline(2.27, linestyle='dashed', color='grey', label='$T_c$')
    elif name == 'Reduced Magnetisation per dipole':
        ax.set_ylabel(f'{name}  [$|m|$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=4, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.axvline(2.27, linestyle='dashed', color='grey', label='$T_c$')
    elif name == 'Spin excess':
        ax.set_ylabel(f'{name} ')
        ax.scatter(T, observable, color='red', zorder=10, label=f'Measured {name} Data')
        ax.axvline(2.27, linestyle='dashed', color='grey', label='$T_c$')
    ax.legend()
    fig.savefig(f'2D {name}  N ={N}.png')

N = 30

sim = IsingModel2D(N, 2)
spins = sim.simulate(100)
sim.visualize_spins(spins[0], 100)


