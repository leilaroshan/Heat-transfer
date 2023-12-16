import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import colorsys

# time steps
ntime_steps = 1000000
# target porosity
target_porosity = 0.4
@jax.jit
def soil_props(porosity):
    n = porosity
    # Soil properties
    lambda_soil = 1.5
    cp_soil = 2000
    rhoS = 1850
    permeability = 1e-16
    # Thermal properties of the soil-water medium
    lambda_medium = lambda_soil * (1 - n)
    cp = cp_soil * (1 - n)
    # Backfill properties
    lambda_backfill = 1.0
    cp_backfill = 800
    rho_backfill = 1900
    d50_backfill = 0.025 * 0.001
    alpha_backfill = lambda_backfill / (rho_backfill * cp_backfill)
    permeability_backfill = 1.0E-8
    target_alpha = lambda_medium / (rhoS * cp)
    target_permeability = 1e-16  # Assuming the same permeability as the soil
    return target_alpha, target_permeability, alpha_backfill, permeability_backfill


@jax.jit
def conduction_convection(porosity, permeability_soil, alpha_soil, permeability_backfill, alpha_backfill):
    # box size, m
    w = h = 1
    L = 1.0
    Nx = 200
    # intervals in x-, y- directions, m
    dx = dy = w / Nx
    # rho water kg/m3
    rhow = 980
    # Viscosity kg/m-s
    mu = 1.00E-03
    # gravity
    g = 9.81  # m/s2
    # Thermal expansion
    beta = 8.80E-05
    # Set conduction to 0 to disable
    conduction = 1.
    convection = 1.
    # Temperature of the cable
    Tcool, Thot = 0, 30
    # pipe geometry
    pr, px, py = 0.025, 0.5, 0.5      # Radius, X position and Y position
    pr2 = pr**2
    # Calculations
    nx, ny = int(w / dx), int(h / dy)
    dx2, dy2 = dx * dx, dy * dy
    # Time step
    dt = 0.01
    # nsteps
    nsteps = ntime_steps
    # Compute heat flow based on permeability
    u0 = jnp.zeros((nx, ny))
    # u0 = jnp.full((nx, ny), Thot)
    mask_cable = np.zeros((nx, ny))
    # offset for second cable
    # offset = pr*1       # Difference = 2*radius of the cable (touching)=1D
    # offset = pr*2           # Difference = 2D
    offset = pr*4          # Difference = 4D
  
    # Cable mask creation loop
    for i in range(nx):
        for j in range(ny):
            x = i * dx
            y = j * dy
            if ((x - px)**2 + (y - py - offset)**2) <= pr2 or ((x - px)**2 + (y - py + offset)**2) <= pr2:
                mask_cable[i, j] = 1.0
    # for i in range(nx):
    #     for j in range(ny):
    #         x = i * dx
    #         y = j * dy
    #         if ((x - px)**2 + (y - py - offset)**2) <= pr2 or ((x - px)**2 + (y - py + offset)**2) <= pr2:
    #             mask_cable[i, j] = 1.0
    mask_cable = jnp.asarray(mask_cable)
    u0 = mask_cable * Thot
    mask_cable_transform = 1 - mask_cable

    # Mask for the backfill material
    mask_backfill = np.zeros((nx, ny))
    backfill_size = 0.5  # size of the backfill box
    for i in range(int((nx//2 - backfill_size/dx/2)), int((nx//2 + backfill_size/dx/2))):
        for j in range(int((ny//2 - backfill_size/dy/2)), int((ny//2 + backfill_size/dy/2))):
            mask_backfill[i, j] = 1.0
    mask_backfill = jnp.asarray(mask_backfill)
    mask_soil = 1 - mask_backfill

    # Update thermal properties for the backfill region
    alpha = jnp.where(mask_backfill, alpha_backfill, alpha_soil)
    permeability = jnp.where(mask_backfill, permeability_backfill, permeability_soil)

    # Apply zero temp at boundaries
    mask_boundaries = np.ones((nx, ny))
    mask_boundaries[:, 0] = 0.0
    mask_boundaries[:, nx-1] = 0.0
    mask_boundaries[0, :] = 0.0
    mask_boundaries[nx-1, :] = 0.0

    # Restrict heat dissipation within the inner box
    # mask boundries, alpha and permeability as an array
    mask_boundaries = jnp.where(mask_backfill, mask_boundaries, 1.0)
    u0 = jnp.multiply(mask_boundaries, u0)

    # Copy to u (u is temperature here)
    u = u0

    # convection_factor
    convection_factor = convection * dt * permeability * (1 / (porosity * mu) * g * rhow) / dy
    #convection_factor = 0
    def step(i, carry):
        u0, u, alpha, permeability = carry
        uip = jnp.roll(u0, 1, axis=0)
        ujp = jnp.roll(u0, 1, axis=1)
        uin = jnp.roll(u0, -1, axis=0)
        ujn = jnp.roll(u0, -1, axis=1)

        # Apply conduction term only within the box
        conduction_term = jnp.where(mask_cable_transform, (uin - 2 * u0 + uip) / dy2 + (ujn - 2 * u0 + ujp) / dx2, 0)
        u = u0 + conduction * dt * alpha * conduction_term + (uip - u0) * convection_factor * (1 - beta * u0)

        # Apply initial conditions and restrict values outside the box to zero
        u = jnp.multiply(u, mask_cable_transform) + mask_cable * Thot
        u = jnp.multiply(u, mask_boundaries)

        # Set u0 as u
        u0 = u
        return (u0, u, alpha, permeability)

    # Iterate
    u0, u, _, _ = jax.lax.fori_loop(0, nsteps, step, (u0, u, alpha, permeability))

    return u, convection_factor, mask_backfill




# Plot
cmap = plt.cm.get_cmap("jet")
def man_cmap(cmap, value=1.):
    colors = cmap(np.arange(cmap.N))
    hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:, :3]])
    hls[:, 1] *= value
    rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0, 1)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)

## Forward heat transfer
porosity = 0.45
alpha_soil, permeability_soil, alpha_backfill, permeability_backfill = soil_props(target_porosity)
u, cf, mask_backfill = conduction_convection(porosity, permeability_soil, alpha_soil, permeability_backfill, alpha_backfill)



# Determine dimensions of the array 'u'
ny, nx = u.shape  # Assuming u is a 2D array

# Create the ax object
fig, ax = plt.subplots()
pcm = ax.pcolormesh(u, cmap=man_cmap(cmap, 1.25), vmin=0, vmax=30)

# Highlight the box inside with another color
box = patches.Rectangle(
    (nx // 4, ny // 4),  # Position of the lower-left corner of the rectangle
    nx // 2, ny // 2,  # Width and height of the rectangle
    linewidth=2, edgecolor='yellow', facecolor='none', label='Backfill Box'
)
ax.add_patch(box)
'''
box_start_x = nx // 4
box_end_x = nx // 4 + nx // 2
box_start_y = ny // 4
box_end_y = ny // 4 + ny // 2
temperature_inside_box = u[box_start_x:box_end_x, box_start_y:box_end_y]
num_elements_inside_box = (box_end_x - box_start_x) * (box_end_y - box_start_y)'''
#print ("num_elements_inside_box", num_elements_inside_box)

# Add an empty list to store average temperatures
average_temps_inside_box = []

# Temp inside the box
'''for step in range(ntime_steps):
    # Calculate average temperature inside the box
    average_temperature_inside_box = jnp.mean(temperature_inside_box)
    average_temps_inside_box.append(average_temperature_inside_box)

# After the loop
average_temps_inside_box = jnp.array(average_temps_inside_box)

# Print the array of average temperatures
print("Array of average temperatures inside the box:")
# print(average_temps_inside_box)
print("temperature_inside_box:", temperature_inside_box[:, :])
average_temperature_inside_box = jnp.mean(temperature_inside_box)
print("Average Temperature inside the box:", average_temperature_inside_box)

# Find the maximum temperature inside the box
max_temperature_inside_box = jnp.max(temperature_inside_box)
print("Max Temperature inside the box:", max_temperature_inside_box)
min_temperature_inside_box = jnp.min(temperature_inside_box)
print("Min Temperature inside the box:", min_temperature_inside_box)
# average_temperature_inside_box = jnp.mean(temperature_inside_box)
# print("Average Temperature inside the box:", average_temperature_inside_box)
'''

# Set x and y ticks as specified
ax.set_xticks([0, 40, 80, 120, 160, 200])
ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 40, 80, 120, 160, 200])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_xlabel('L (m)')
ax.set_ylabel('H (m)')

# Create a ScalarMappable for colorbar
sm = plt.cm.ScalarMappable(cmap=man_cmap(cmap, 1.25))
sm.set_array(u)
# plt.colorbar(sm, ticks=[0,5,10,15,20,25,30])
cbar = plt.colorbar(sm, ticks=[0, 5, 10, 15, 20, 25, 30])

# Add legend
plt.legend()
plt.show()
plt.imshow(cf)
plt.colorbar()



average_temperature_inside_box = jnp.mean(u[mask_backfill.astype(bool)])
max_temperature_inside_box = jnp.max(u[mask_backfill.astype(bool)])
min_temperature_inside_box = jnp.min(u[mask_backfill.astype(bool)])
print('avg: ',average_temperature_inside_box)
print('min: ',min_temperature_inside_box)
print('max: ',max_temperature_inside_box)





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
