import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math as math

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

class ID_IsingModel:
    """
    Class to create 1-dimensional ising model
    """
    def __init__(self, N, T):
        """
        Initialises state variables

        N = number of spins
        T = temperature
        epsilon = energy of individual dipole
        """
        self.N = N
        self.T = T
        self.k = 1
        self.mu = 1
        self.beta = 1 / (self.k * self.T)
        self.spins = np.random.choice([-1, 1], size=N)
        self.epsilon = 1
        self.u = -self.epsilon * math.tanh(self.epsilon*self.beta)
        self.S = (self.epsilon/self.T)*(1-math.tanh(self.beta*self.epsilon))+self.k*math.log(1+ math.exp(-2*self.epsilon*self.beta))
        self.c = (self.epsilon**2*self.beta)/(self.T*math.cosh(self.beta*self.epsilon)**2)
        self.f = self.u - self.T * self.S
    
    def get_energy(self, config):
        """
        Method to calculate the net energy of the system with periodic boundary conditions.
        returns float 
        """
        energy = 0
        N = len(config)
        
        for i in range(N):
            interaction = config[i] * config[(i + 1) % N]
            energy += interaction

        #periodic boundary condition:
        energy += config[0] * config[N - 1]

        return -self.epsilon * energy / self.N
    
    def get_free_energy(self, energy, entropy, i):
        """
        Method to get the free energy of the system
        returns float
        """
        free_energy = energy[i] - self.T*entropy[i]
        return free_energy 
    
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
            if config[i] == 1:
                up += 1
            if config[i] == -1:
                down += 1

        s = (up - down)/2

        entropy = self.k*(-(2*s**2)/self.N + 0.5*math.log(2/(math.pi*self.N)) + self.N*math.log(2))
    
        return entropy / self.N 
    
    def get_heat_capacity(self, energy):
        """
        Method to get the heat capacity of the system
        return float
        """
        return -(np.std(energy)**2)/(self.N*self.k*self.T**2)
    
    def get_magnet(self, spins):
        """
        Method to get the reduced net magnetisation
        """
        return np.average(spins)

    def metropolis(self):
        for i in range(self.N):
            index = random.randint(0, self.N - 1)
            old_spin = self.spins[index]
            new_spin = -old_spin
            delta_energy = 2 * old_spin * (
                    self.spins[(index + 1) % self.N] + self.spins[(index - 1) % self.N])

            if delta_energy < 0 or random.random() < np.exp(
                    -delta_energy / (self.k * self.T)):
                self.spins[index] = new_spin
 
    def get_spins(self):
        """
        Method to return the configuration of spins
        """
        return self.spins
    
    def simulate(self, num_iterations):
        """
        Method to complete the simulation

        returns: all data required for plotting
        """
        boxes = []
        boxes.append(self.spins.copy())
        energy = np.zeros(num_iterations)
        free_energy = np.zeros(num_iterations)
        entropy = np.zeros(num_iterations)
        magnet = np.zeros(num_iterations)
        heat_capacity = np.zeros(num_iterations)

        #Bring state to equilibrium first
        for i in range(0,1024):
            self.metropolis()

        for i in range(0,num_iterations):
            self.metropolis()
            boxes.append(self.get_spins().copy())
            energy[i] = self.get_energy(self.get_spins())
            entropy[i] = self.get_entropy(self.get_spins())
            heat_capacity[i] = np.var(energy)
            free_energy[i] = self.get_free_energy(energy, entropy, i)
            magnet[i] = self.get_magnet(self.spins)
        return boxes, energy, entropy, heat_capacity, free_energy, magnet

    def visualise_spins(self, boxes, num_iterations):
        """
        Method to create animation to display the state of the system
        """
        fig, ax = plt.subplots(figsize=(8, 2))

        def update(frame):
            ax.clear()
            ax.set_xlim(0, self.N)
            ax.set_ylim(-0, 0.5)
            ax.imshow(np.array([boxes[frame]]), cmap='gray', aspect=15)
            ax.set_title(f'Iteration: {frame + 1}/{num_iterations} ', fontdict=font)
            ax.set_xlabel(f'T = {self.T}')
            if frame == num_iterations - 1:
                fig.savefig('Final State.png')
            if frame == 0:
                fig.savefig('Initial State.png')
            return

        ani = FuncAnimation(fig, update, frames = num_iterations, interval=50)
        plt.show()

    def visualise_energy(self, energy, num_iterations):
        """
        Method to create an animated plot of the energy of the system over the course of the simulation
        """
        fig, ax = plt.subplots()
        

        def update(frame):
            x = np.arange(0,frame,1)
            ax.clear()
            ax.set_xlabel("Number of Monte Carlo Iterations")
            ax.set_ylabel("Energy per Spin")
            ax.plot(x,energy[:frame ])
            ax.axhline(self.u, color='red')

            ax.set_title(f'Iteration: {frame + 1}/{num_iterations}', fontdict=font)
            if frame == num_iterations - 1:
                fig.savefig('Energy Graph.png')
            return 
        
        
        
        ani = FuncAnimation(fig, update, frames = num_iterations, interval = 10)
        plt.show()

    def visualise_magnet(self, magnet, num_iterations):
        """
        Method to create an animated plot of the energy of the system over the course of the simulation
        """
        fig, ax = plt.subplots()
        

        def update(frame):
            x = np.arange(0,frame,1)
            ax.clear()
            ax.set_xlabel("Number of Monte Carlo Iterations")
            ax.set_ylabel("Reduced net magnetisation")
            ax.plot(x,magnet[:frame ])

            ax.set_title(f'Iteration: {frame + 1}/{num_iterations}', fontdict=font)
            if frame == num_iterations - 2:
                fig.savefig('Reduced Magnetisation Graph.png')
            return 
        
        ani = FuncAnimation(fig, update, frames = num_iterations, interval = 10)
        plt.show()

    def visualise_free_energy(self, free_energy, num_iterations):
        """
        Method to create an animated plot of the free energy of the system over the course of the simulation
        """
        fig, ax = plt.subplots()

        def update(frame):
            x = np.arange(0,frame,1)
            ax.clear()
            ax.set_xlabel("Number of Monte Carlo Iterations")
            ax.set_ylabel("Free energy per spin")
            ax.plot(x,free_energy[:frame ])
            ax.axhline(self.f, color='red')

            ax.set_title(f'Iteration: {frame + 1}/{num_iterations}', fontdict=font)
            if frame == num_iterations - 1:
                fig.savefig('Free Energy Graph.png')
            return 
        
        ani = FuncAnimation(fig, update, frames = num_iterations, interval = 10)
        plt.show()
        return
    
    def visualise_entropy(self, entropy, num_iterations):
        """
        Method to create an animated plot of the entropy of the system over the course of the simulation
        """
        fig, ax = plt.subplots()

        def update(frame):
            x = np.arange(0,frame,1)
            ax.clear()
            ax.set_xlabel("Number of Monte Carlo Iterations")
            ax.set_ylabel("Entropy per spin")
            ax.plot(x,entropy[:frame ])
            ax.axhline(self.S, color='red')

            ax.set_title(f'Iteration: {frame + 1}/{num_iterations}', fontdict=font)
            if frame == num_iterations - 1:
                fig.savefig('Entropy Graph.png')
            return 
        
        ani = FuncAnimation(fig, update, frames = num_iterations, interval = 10)
        plt.show()
        return
    
    def visualise_heat_capacity(self, heat_capacity, num_iterations):
        """
        Method to create an animated plot of the heat capcity of the system over the course of the simulation
        """
        fig, ax = plt.subplots()
        

        def update(frame):
            x = np.arange(0,frame,1)
            ax.clear()
            ax.set_xlabel("Number of Monte Carlo Iterations")
            ax.set_ylabel("Heat capacity per spin")
            ax.plot(x,heat_capacity[:frame ]/self.N)
            ax.axhline(self.c, color='red')

            ax.set_title(f'Iteration: {frame + 1}/{num_iterations}', fontdict=font)
            if frame == num_iterations - 1:
                fig.savefig('Heat Capacity Graph.png')
            return 
        
        ani = FuncAnimation(fig, update, frames = num_iterations, interval = 1)
        plt.show()
        return

    def plot_final_state(self, observable, num_iterations, name, prediction):
        fig, ax = plt.subplots()
    
        x = np.arange(0,num_iterations,1)
        ax.clear()
        ax.set_xlabel('Number of Monte Carlo Iterations')
        ax.set_ylabel(f'{name} per spin')
        ax.axhline(prediction, color='red')
        ax.set_title(f'Final {name} state')
        ax.plot(x, observable)
        fig.savefig(f'Final {name} State.png')

    def compare_observables(self, energy, entropy, heat_capacity, free_energy, magnet):
        mean_energy = round(np.average(energy),3)
        mean_entropy = round(np.average(entropy),3)
        mean_heat_capcity = round(self.N*np.var(energy)/self.T**2,3)
        mean_free_energy = round(np.average(free_energy), 3)
        mean_magnetisation = round(np.average(magnet))
        delta_energy = round(np.std(energy),3)
        delta_entropy = round(np.std(entropy),3)
        delta_heat_capacity = round(np.std(heat_capacity),3)
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

        return mean_energy, delta_energy, mean_entropy, delta_entropy, mean_heat_capcity, delta_heat_capacity, mean_free_energy, delta_free_energy, mean_magnetisation, delta_magnetisation

def plot_observable_t(observable, delta, T, name):
    fig, ax= plt.subplots()
    ax.clear()
    ax.set_xlabel(f'Temperature  [$\epsilon / k$]')
    ax.set_title(f'{name} vs Temperature')
    ax.grid()
    
    tp = np.arange(0.1,1,0.005)
    #energy
    energy = [-math.tanh(1/x) for x in tp]
    #entropy
    entropy = [((1/x)*(1-math.tanh(1/x)) + math.log(1+math.exp(-2/x))) for x in tp]
    #heat capacity
    heat_capacity = [(1/(x**2*math.cosh(1/x)**2)) for x in tp]
    #free energy
    free_energy = [(-1-x*math.log(1+math.exp(-2/x))) for x in tp]

    if name == 'Energy per dipole':
        ax.set_ylabel(f'{name}  [$\epsilon$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.plot(tp,energy,label='Predicted Energy', color='royalblue')
    elif name == 'Entropy per dipole':
        ax.set_ylabel(f'{name}  [$k$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.plot(tp,entropy, label='Predicted Entropy', color='royalblue')
    elif name == 'Heat Capacity per dipole':
        ax.set_ylabel(f'{name}  [$k$]')
        ax.errorbar(T[1:], observable[1:], delta[1:], color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.plot(tp,heat_capacity,label='Predicted Heat Capacity', color='royalblue')
    elif name == 'Free Energy per dipole':
        ax.set_ylabel(f'{name}  [$\epsilon$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=4, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
        ax.plot(tp,free_energy, label='Free energy', color='royalblue')
    elif name == 'Reduced Magnetisation per dipole':
        ax.set_ylabel(f'{name}')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=4, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')


    ax.legend()
    plt.show()

#plot_observable_t(energies, delta_energies, T, 'Energy per dipole')
#plot_observable_t(entropys, delta_entropies, T, 'Entropy per dipole')
#plot_observable_t(heat_capacitys, delta_heat_capacities, T, 'Heat Capacity per dipole')
#plot_observable_t(free_energys, delta_free_energies, T, 'Free Energy per dipole')
#plot_observable_t(magnetisations,delta_magnets, T, 'Reduced Magnetisation per dipole')

for i, T in enumerate(magnetisations):
    temperature = i + 1  # You can use your actual temperature values here
    plt.hist(T, bins=10, label=f'Temperature {temperature}', alpha=0.5)

plt.xlabel('Magnetization')
plt.ylabel('Frequency')
plt.title('Histogram of Reduced Magnetisation at Different Temperatures')
plt.legend()
plt.show()