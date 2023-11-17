import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math as math
from numba import jit
from scipy.optimize import curve_fit

@jit(nopython=True)
def metropolis_core(spins, N, k, T):
    """
    Method to apply the metropolis algorithm
    """
    for _ in range(0, 10*N*N*N):
        i, j, m = rnd.randint(0, N - 1), rnd.randint(0, N - 1), rnd.randint(0, N - 1)
        old_spin = spins[i, j, m]
        new_spin = -old_spin
        delta_energy = 2 * old_spin * (
            spins[(i + 1) % N, j, m] + spins[(i - 1) % N, j, m] +
            spins[i, (j + 1) % N, m] + spins[i, (j - 1) % N, m] +
            spins[i, j, (m + 1) % N] + spins[i, j, (m - 1) % N]
        )
        if delta_energy <= 0 or rnd.random() < np.exp(
                -delta_energy / (k * T)):
            spins[i][j][m] = new_spin

@jit(nopython=True)
def energy_core(config, N):
    """
    Calculate the energy of the system in a particular state
    """
    energy = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                interaction = config[i][j][k]*(
                    config[(i + 1) % N][j][k] +
                    config[i][(j + 1) % N][k] +
                    config[i][j][(k + 1) % N]
                )
                energy += interaction
    return -energy / N**3

class IsingModel3D:
    """
    Class for 3D Ising Model
    """
    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.spins = np.random.choice([-1, 1], size=(N, N, N), p=(0.75,0.25))  # 3D lattice
        self.k = 1
        self.beta = 1 / (self.k * self.T)
        self.epsilon = 1
    
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
                for k in range(len(config)):
                    if config[i][j][k] == 1:
                        up += 1
                    if config[i][j][k] == -1:
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
        up = np.sum(config == 1)
        down = np.sum(config == -1)
        s = (up - down) / 2
        return s / self.N**2

    def metropolis(self):
        metropolis_core(self.spins, self.N, self.k, self.T)

    def get_energy(self):
        energy = energy_core(self.spins, self.N)
        return energy

    def simulate(self, num_iterations):
        """
        Method to complete the simulation
        """
        spins_history = []
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
            # if i < num_iterations/2:
            #     self.T += 0.01
            # elif i > num_iterations/2:
            #     self.T -= 0.01

            self.metropolis()
            print(f"Current step: {i}")
            spins_history.append(self.spins.copy())
            energy = self.get_energy()
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
        Method to animate the evolution of the system in 3D
        """
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            ax.set_title(f'Iteration: {frame + 1} / {len(spins_history) - 1}', fontsize=16)
            ax.set_xlabel(f'T = {self.T}')
            
            # Extract pins for the current frame
            current_spins = spins_history[frame]
            
            # Get the coordinates of pins
            x_up, y_up, z_up = [], [], []
            x_down, y_down, z_down = [], [], []

            for x in range(self.N):
                for y in range(self.N):
                    for z in range(self.N):
                        if current_spins[x, y, z] == 1:  # Use square brackets for indexing
                            x_up.append(x)
                            y_up.append(y)
                            z_up.append(z)
                        elif current_spins[x, y, z] == -1:  # Use square brackets for indexing
                            x_down.append(x)
                            y_down.append(y)
                            z_down.append(z)

            ax.scatter(x_up, y_up, z_up, c='b', marker='o', s=50, label='Up Spin')
            ax.scatter(x_down, y_down, z_down, c='r', marker='o', s=50, label='Down Spin')
            
            ax.legend()

            if frame == 0:
                fig.savefig(f'3D_initial_state_T={self.T}.png')
            if frame == num_iterations / 4:
                fig.savefig(f'3D_Ising_q1_point_T={self.T}.png')
            if frame == num_iterations / 2:
                fig.savefig(f'3D_Ising_midpoint_T={self.T}.png')
            if frame == 3 * num_iterations / 4:
                fig.savefig(f'3D_Ising_q2_point_T={self.T}.png')
            if frame == num_iterations - 1:
                fig.savefig(f'3D_final_state_T={self.T}.png')

        ani = FuncAnimation(fig, update, frames=num_iterations, interval=50)
        plt.show()

    def create_3D_animation(self, spins_history, num_iterations):
        """
        Method to animate the evolution of the system in 3D
        """
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            ax.set_title(f'Iteration: {frame + 1} / {len(spins_history) - 1}', fontsize=16)
            
            current_spins = spins_history[frame]
            
            x_up, y_up, z_up = np.where(current_spins == 1)
            x_down, y_down, z_down = np.where(current_spins == -1)

            ax.scatter(x_up, y_up, z_up, c='b', marker='o', s=50, label='Up Spin')
            ax.scatter(x_down, y_down, z_down, c='r', marker='o', s=50, label='Down Spin')
            
            ax.legend()

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

        # print(self.T)
        # print(f'Measured Energy: {mean_energy} +/- {delta_energy}')
        # print(f'Predicted Energy: {round(self.u,3)}')
        # print(f'Measured Entropy: {mean_entropy} +/- {delta_entropy}')
        # print(f'Predicted Entropyy: {round(self.S,3)}')
        # print(f'Measured Heat Capcity: {mean_heat_capcity} +/- {delta_heat_capacity}')
        # print(f'Predicted Heat Capcity: {round(self.c,3)}')
        # print(f'Measured Free Energy: {mean_free_energy} +/- {delta_free_energy}')
        #print(f'Predicted Free Energy: {round(self.f,3)}')
        #print(f'Measured Reduced Magnetisation: {mean_magnetisation} +/- {delta_magnetisation}')

        return mean_energy, delta_energy, mean_entropy, delta_entropy, mean_heat_capcity, delta_heat_capacity, mean_free_energy, delta_free_energy, mean_magnetisation, delta_magnetisation, mean_spin_excess


def plot_observable_t(observable, delta, T, name):
    fig, ax= plt.subplots()
    ax.clear()
    ax.set_xlabel(f'Temperature  [$\epsilon / k$]')
    ax.set_title(f'{name} vs Temperature')
    ax.grid()

    if name == 'Energy per dipole':
        ax.set_ylabel(f'{name}  [$\epsilon$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
    elif name == 'Entropy per dipole':
        ax.set_ylabel(f'{name}  [$k$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
    elif name == 'Heat Capacity per dipole':
        ax.set_ylabel(f'{name}  [$k$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=5, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
    elif name == 'Free Energy per dipole':
        ax.set_ylabel(f'{name}  [$\epsilon$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=4, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
    elif name == 'Reduced Magnetisation per dipole':
        ax.set_ylabel(f'{name}  [$|m|$]')
        ax.errorbar(T, observable, delta, color='red', elinewidth=1.5, barsabove='true', capsize=4, fmt='o', markersize=3, capthick=1, zorder=10, label=f'Measured {name} Data')
    elif name == 'Spin excess':
        ax.set_ylabel(f'{name} ')
        ax.scatter(T, observable, color='red', zorder=10, label=f'Measured {name} Data')
    elif name == 'Magnetic Susceptibility':
        ax.set_ylabel(f'{name} [m/kK]')
        ax.scatter(T, observable, color='red', zorder=10, label=f'Measured {name} Data')
    ax.legend()
    fig.savefig(f'3D {name}.png')

def get_mag_susc(magnetisation_data:list, temperature: float) -> list:
    squared_mag = [x**2 for x in magnetisation_data]
    square_ave = np.average(squared_mag)
    ave_squared = np.average(magnetisation_data)**2

    return (square_ave - ave_squared)/temperature

def lorentzian(T, A, Tc, B, C):
    return A / ((T - Tc)**2 + B) + C

def binder_cumulant(magnetisation_data):
    M2 = np.average(np.square(magnetisation_data))
    M4 = np.average(np.power(magnetisation_data, 4))
    U4 = 1 - M4 / (3 * M2**2)
    return U4

def polynomial_func(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

T = np.linspace(3, 7, 15)
energy_data = []
del_energy = []
entropy_data = []
del_entropy = []
heat_capacity_data = []
del_heat_capacity = []
free_energy_data = []
del_free_energy = []
magnetisation_data = []
del_magnetisation = []
spin_excess_data = []
del_spin_excess = []
mag_susc_data = []
del_mag_susc = []

# for i in range(len(T)):
#     ising = IsingModel3D(25, T[i])
#     data = ising.simulate(700)

#     energy_data.append(np.mean(data[1]))
#     del_energy.append(np.var(data[1]))

#     entropy_data.append(np.mean(data[2]))
#     del_entropy.append(np.var(data[2]))

#     heat_capacity_data.append(np.mean(data[3]))
#     del_heat_capacity.append(np.var(data[3]))


#     free_energy_data.append(np.mean(data[4]))
#     del_free_energy.append(np.var(data[4]))


#     magnetisation_data.append(np.mean(data[5]))
#     del_magnetisation.append(np.var(data[5]))

#     spin_excess_data.append(np.mean(data[6]))
#     del_spin_excess.append(np.var(data[6]))

#     mag_susc_data.append(get_mag_susc(data[5], T[i]))


# plot_observable_t(energy_data, del_energy, T, 'Energy per dipole')
# plot_observable_t(entropy_data, del_entropy, T, 'Entropy per dipole')
# plot_observable_t(heat_capacity_data, del_heat_capacity, T, 'Heat Capacity per dipole')
# plot_observable_t(free_energy_data, del_free_energy, T, 'Free Energy per dipole')
# plot_observable_t(magnetisation_data, del_magnetisation, T, 'Reduced Magnetisation per dipole')
# plot_observable_t(spin_excess_data, del_spin_excess, T, 'Spin excess')
# plot_observable_t(mag_susc_data, del_mag_susc, T, 'Magnetic Susceptibility')

# popt, pcov = curve_fit(lorentzian, T, mag_susc_data, p0=(0.0005,4.5,0.01, 0))
# perr = np.sqrt(np.diag(pcov))

# A_fit, Tc_fit, B_fit, C_fit = popt

# print(f"Estimated critical temperature: {Tc_fit:.3f} +/- {perr}")
# print(f'{A_fit}')
# print(f'{Tc_fit}')
# print(f'{B_fit}')
# print(f'{C_fit}')

# fig, ax = plt.subplots()
# ax.clear()
# ax.scatter(T, mag_susc_data, color='blue', zorder=10, label='Magnetic Suscuptibility Data')
# ax.plot(T, lorentzian(T, *popt), 'r-', label='Lorentzian Fit')
# ax.set_xlabel('Temperature')
# ax.set_ylabel('Magnetic Susceptibility [m/kK]')
# ax.legend()
# ax.set_title('Magnetic Susceptibility vs Temperature with Lorentzian Fit')
# fig.savefig('3D Magnetic Susceptibility with Fit.png')
# plt.show()

lattice_sizes = [15, 20, 25]
binder_data = {}
fit_params = {}

# Simulation and Binder Cumulant Calculation
for L in lattice_sizes:
    U4_values = []
    
    for i in range(len(T)):
        ising = IsingModel3D(L, T[i])
        data = ising.simulate(700)
        
        U4_values.append(binder_cumulant(data[5]))
        
    binder_data[L] = U4_values

    # Curve fitting
    popt, pcov = curve_fit(polynomial_func, T, U4_values)
    perr = np.sqrt(np.diag(pcov))
    fit_params[L] = popt

# Finding the intersection points
intersection_temps = []
for L1, L2 in [(15, 20), (20, 25), (15, 25)]:
    # Here we will solve for intersection using a numerical method, you can use more sophisticated methods if required
    for t in np.linspace(min(T), max(T), 10000): # Fine-grained temperature values
        if abs(polynomial_func(t, *fit_params[L1]) - polynomial_func(t, *fit_params[L2])) < 0.001: # Tolerance
            intersection_temps.append(t)
            break

Tc_estimate = np.mean(intersection_temps)

# Plotting
fig, ax = plt.subplots()
for L, U4_values in binder_data.items():
    ax.scatter(T, U4_values, label=f'Lattice size {L}^3')
    ax.plot(T, polynomial_func(T, *fit_params[L]), '--', label=f'Fit for {L}^3') 

#ax.axvline(Tc_estimate, color='black', linestyle=':', label=f'Estimated Tc = {Tc_estimate:.3f}')
plt.xlabel('Temperature')
plt.ylabel('Binder Cumulant \(U_4\)')
plt.legend()
plt.title('Binder Cumulant vs Temperature for different lattice sizes')
plt.savefig('#D Binder Cumulant.png')
plt.show()

print(f"Estimated Critical Temperature: {Tc_estimate} +/- {perr}")