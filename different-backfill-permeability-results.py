import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import colorsys
import matplotlib.patches as patches

# Load 'u' array from the CSV file
u = np.loadtxt('output_u.csv', delimiter=',')

# Load 'cf' array from the CSV file
cf = np.loadtxt('output_cf.csv', delimiter=',')

# Assuming the mask_backfill array is generated or defined in this code
# You can add the code to define or generate it before calculating statistics

# Calculate statistics using the loaded 'u' array and 'mask_backfill'
average_temperature_inside_box = jnp.mean(u[mask_backfill.astype(bool)])
max_temperature_inside_box = jnp.max(u[mask_backfill.astype(bool)])
min_temperature_inside_box = jnp.min(u[mask_backfill.astype(bool)])
print('avg: ', average_temperature_inside_box)
print('min: ', min_temperature_inside_box)
print('max: ', max_temperature_inside_box)

# Data for line 1
spacing_1 = 0.5
perm_1 = [1e-8, 1e-11, 1e-15]
avg_temp_1 = [3.753, 4.277, 11.354]

# Data for line 2
spacing_2 = 0.1
perm_2 = [1e-8, 1e-11, 1e-15]
avg_temp_2 = [3.883, 4.703, 12.576]

# Data for line 3
spacing_3 = 0.2
perm_3 = [1e-8, 1e-11, 1e-15]
avg_temp_3 = [4.07, 4.949, 14.145]

# Plotting the data
plt.figure(figsize=(10, 6))  # Set figure size

plt.plot(perm_1, avg_temp_1, marker='o', linestyle='-', label=f'Spacing = {spacing_1}m , 1D', color='red')
plt.plot(perm_2, avg_temp_2, marker='s', linestyle='-', label=f'Spacing = {spacing_2}m , 2D', color='green')
plt.plot(perm_3, avg_temp_3, marker='^', linestyle='-', label=f'Spacing = {spacing_3}m , 4D', color='blue')

# Adding labels and legend
plt.xlabel('Permeability (m²) (log scale)')
plt.ylabel('Average Temperature')
plt.title('Average Temperature vs Permeability for Different Spacings')
plt.xscale('log')  # Use log scale for x-axis
plt.ylim(0, 15)  # Set y-axis limits
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

# Displaying legend
plt.legend()
# Display the plot
plt.show()
