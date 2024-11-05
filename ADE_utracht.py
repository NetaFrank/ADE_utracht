import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
import os
import pandas as pd
from matplotlib.animation import FFMpegWriter
import numpy as np
from numba import njit


def define_properties(Ny, Nx, load_data):
    # Create random heterogeneous fields for K and phi
    if load_data == 1:
        # Load K_matrix from a CSV file
        k_matrix = pd.read_csv(
            r"C:\Users\neta\OneDrive - Technion\Desktop\Technion\MSC\ADE_yaniv\OneDrive_1_11-2-2024\k_matrix.csv", header=None).values

        phi_matrix = pd.read_csv(
            r"C:\Users\neta\OneDrive - Technion\Desktop\Technion\MSC\ADE_yaniv\OneDrive_1_11-2-2024\phi_matrix.csv",
            header=None).values

        # Calculate the mean of phi matrix
        phi_mean = np.nanmean(phi_matrix)

    else:
        np.random.seed(1)  # For reproducibility
        k_mean = 0.06  # Mean hydraulic conductivity [cm/min]
        k_std = 0.1 * k_mean  # Standard deviation
        phi_mean = 0.35  # Mean porosity [-]
        phi_std = 0.1  # Standard deviation

        # Generate spatially varying K and phi with dimensions (Ny + 1, Nx + 1)
        k_matrix = k_mean + k_std * np.random.randn(Ny + 1, Nx + 1)
        k_matrix[k_matrix <= 0] = k_mean  # Ensure K is positive

        phi_matrix = phi_mean + phi_std * np.random.randn(Ny + 1, Nx + 1)

    # Ensure K is positive by replacing negative values with the minimum value in K_matrix
    k_matrix[k_matrix <= 0] = np.min(k_matrix)

    # Ensure phi values are realistic (0 < phi < 1)
    phi_matrix[phi_matrix <= 0] = phi_mean
    phi_matrix[phi_matrix >= 1] = phi_mean

    return k_matrix, phi_matrix


@njit
def flow_solver(Lx, Ly, Nx, Ny, head_diff, k_matrix):
    # Parameters
    h_in = head_diff  # Hydraulic head at inlet [m]
    h_out = 0  # Hydraulic head at outlet [m]

    # Grid spacing
    dx = Lx / Nx
    dy = Ly / Ny

    # Initialize hydraulic head matrix
    h = np.zeros((Ny + 1, Nx + 1))  # Rows: y, Columns: x

    # Initial guess for h (linear gradient)
    for i in range(Nx + 1):
        h[:, i] = h_in - (h_in - h_out) * i / Nx

    # Apply k_diff_norm on h for initial guess adjustment (align with MATLAB code)
    k_diff = k_matrix[:-1, :] - k_matrix[1:, :]
    k_diff_avg = np.sum(k_matrix[:-1, :], axis=0) * np.ones(k_diff.shape)
    k_diff_norm = 1 + k_diff / k_diff_avg
    h[1:, :] = h[1:, :] * k_diff_norm

    # Boundary conditions
    h[:, 0] = h_in  # Left boundary (x = 0)
    h[:, -1] = h_out  # Right boundary (x = L)

    # Iterative solution using Successive Over-Relaxation (SOR)
    tol = 1e-6
    max_iter = 10000
    omega = 1.5

    for iter in range(max_iter):
        h_old = h.copy()

        for j in range(1, Ny):
            for i in range(1, Nx):
                # Compute average K values at interfaces
                k_e = (k_matrix[j, i] + k_matrix[j, i + 1]) / 2
                k_w = (k_matrix[j, i] + k_matrix[j, i - 1]) / 2
                k_n = (k_matrix[j, i] + k_matrix[j - 1, i]) / 2
                k_s = (k_matrix[j, i] + k_matrix[j + 1, i]) / 2

                # Coefficients
                a_e = k_e / dx ** 2
                a_w = k_w / dx ** 2
                a_n = k_n / dy ** 2
                a_s = k_s / dy ** 2
                a_p = a_e + a_w + a_n + a_s

                # Update hydraulic head
                h_new = (a_e * h[j, i + 1] + a_w * h[j, i - 1] +
                         a_n * h[j - 1, i] + a_s * h[j + 1, i]) / a_p

                # SOR Update
                h[j, i] = (1 - omega) * h[j, i] + omega * h_new

        # Apply boundary conditions after each iteration
        h[:, 0] = h_in  # Left boundary (x = 0)
        h[:, -1] = h_out  # Right boundary (x = L)
        h[0, :] = h[1, :]  # Bottom boundary (y = 0), no-flow
        h[-1, :] = h[-2, :]  # Top boundary (y = W), no-flow

        # Check convergence
        max_diff = np.max(np.abs(h - h_old))
        if max_diff < tol:
            print(f'Hydraulic head converged in {iter} iterations.')
            break
    else:
        print('Warning: Hydraulic head did not converge within the maximum number of iterations.')

    # Compute gradients
    dh_dx = np.zeros_like(h)
    dh_dy = np.zeros_like(h)
    for j in range(1, Ny):
        for i in range(1, Nx):
            dh_dx[j, i] = (h[j, i + 1] - h[j, i - 1]) / (2 * dx)
            dh_dy[j, i] = (h[j + 1, i] - h[j - 1, i]) / (2 * dy)

    # Compute velocities
    v_x = -k_matrix * dh_dx  # Velocity in x-direction
    v_y = -k_matrix * dh_dy  # Velocity in y-direction

    return h, v_x, v_y


def calc(Lx, Ly, T, vx, vy, D, Nx, Ny, K,
         num_snapshots):  # Calculating step sizes as a func. of full length and num of steps
    dx = Lx / Nx  # Spatial step size in x [cm]
    dy = Ly / Ny  # Spatial step size in y [cm]
    dt = T / K  # Time step size [min]

    # Initialize two matrices for current and next time steps
    c_current = np.zeros((Nx + 2, Ny + 2))  # larger domain for numeric solution
    c_next = np.zeros_like(c_current)  # for the next time step

    # Initial condition - everything is zero

    c_current[:, 1] = c_inlet  # Initial condition [mg/cm^2] - total mass divided by sponge area

    # Coefficients
    alpha_x = np.full_like(c_current, (vx * dt / (2 * dx)))
    alpha_y = np.full_like(c_current, (vy * dt / (2 * dy)))
    gamma_x = np.full_like(c_current, (D * dt / dx ** 2))
    gamma_y = np.full_like(c_current, (D * dt / dx ** 2)) + 0.1 * gamma_x

    # Preallocate a 3D array to store every 10th c_current
    c_matrix = np.zeros((num_snapshots, Nx + 2, Ny + 2))  # Preallocate 3D array for snapshots

    # Time-stepping with Euler methodS

    snapshot_index = 0
    for k in range(K):
        # Stability check (CFL condition)
        assert np.all(alpha_x <= 1), f"Stability condition not met for advection at k = {k}"
        assert np.all(alpha_y <= 1), f"Stability condition not met for advection at k = {k}"
        assert np.all(gamma_x <= 0.5), f"Stability condition not met for diffusion in x at k = {k}"
        assert np.all(gamma_y <= 0.5), f"Stability condition not met for diffusion in y at k = {k}"

        # Calculate advection and diffusion terms
        adv_x = np.zeros_like(c_current)
        adv_y = np.zeros_like(c_current)
        diff_x = np.zeros_like(c_current)
        diff_y = np.zeros_like(c_current)

        # y is not identical to x to match the changing in vertical velocity
        adv_x[1:-1, :] = - alpha_x[1:-1, :] * (c_current[2:, :] - c_current[:-2, :])
        adv_y[:, 1:-1] = - alpha_y[:, 1:-1] * (c_current[:, 2:] - c_current[:, :-2])
        diff_x[1:-1, :] = gamma_x[1:-1, :] * (c_current[2:, :] - 2 * c_current[1:-1, :] + c_current[:-2, :])
        diff_y[:, 1:-1] = gamma_y[:, :-2] * (c_current[:, 2:] - 2 * c_current[:, 1:-1] + c_current[:, :-2])

        c_next[1:-1, 1:-1] = c_current[1:-1, 1:-1] + adv_x[1:-1, 1:-1] + adv_y[1:-1, 1:-1] + diff_x[1:-1,
                                                                                             1:-1] + diff_y[1:-1, 1:-1]

        # Save every num_snapshot'th c_current into c_matrix
        if k % (K / num_snapshots) == 0:
            c_matrix[int(k / (K / num_snapshots))] = c_current

        # Swap matrices: c_next becomes c_current for the next time step
        c_current, c_next = c_next, c_current

        # initializing initial step as a pulse for the first 10 days
        if k * dt < 10:
            c_current[:, 1] = c_inlet

    return c_matrix[:, 1:, 1:]  # remove the first not relevant column and raw


# Function to plot hydraulic head distribution
def plot_hydraulic_head(h, x, y):
    plt.figure()
    plt.imshow(np.log10(np.flipud(h)), extent=[x[0], x[-1], y[0], y[-1]], aspect='auto')
    plt.gca().invert_yaxis()
    plt.colorbar(label='log10 Hydraulic Head')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.title('Hydraulic Head Distribution')
    # plt.axis('equal')
    plt.show()


# Function to plot velocity field
def plot_velocity_field(x, y, v_x, v_y, skip=10):
    # Generate meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Downsample data for clarity in the quiver plot
    X_downsampled = X[::skip, ::skip]
    Y_downsampled = Y[::skip, ::skip]
    v_x_downsampled = v_x[::skip, ::skip]
    v_y_downsampled = v_y[::skip, ::skip]

    # Calculate the velocity magnitude for colormap
    velocity_magnitude = np.sqrt(v_x ** 2 + v_y ** 2)
    velocity_magnitude_downsampled = velocity_magnitude[::skip, ::skip]

    # Create the quiver plot with a jet colormap
    plt.figure()
    quiver_plot = plt.quiver(X_downsampled, Y_downsampled, v_x_downsampled, v_y_downsampled,
                             velocity_magnitude_downsampled, cmap='jet')

    # Add colorbar for the velocity magnitude
    cbar = plt.colorbar(quiver_plot)
    cbar.set_label('Velocity Magnitude [m/s]')

    # Set labels and title
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Velocity Field')
    plt.show()


# Function to plot absolute velocity field
def plot_absolute_velocity(x, y, v_x, v_y):
    # Calculate the absolute velocity magnitude and take log10
    abs_velocity = np.sqrt(v_x ** 2 + v_y ** 2)
    log_abs_velocity = np.log10(abs_velocity)

    # Create the plot
    plt.figure()
    plt.imshow(np.flipud(log_abs_velocity), extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', cmap='jet')

    # Add colorbar for the velocity magnitude
    cbar = plt.colorbar()
    cbar.set_label('Log10 of Absolute Velocity Magnitude')

    # Set labels and title
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Abs Velocity Field')
    plt.show()


def plot_vy_velocity_field(x, y, v_y):

    # Take the absolute values of v_y and apply log10 scaling
    abs_vy = np.abs(v_y)
    log_vy = np.log10(abs_vy + 1e-10)  # Small offset to avoid log of zero

    # Create the plot
    plt.figure()
    plt.imshow(np.flipud(log_vy), extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', cmap='jet')

    # Add colorbar for the v_y magnitude
    cbar = plt.colorbar()
    cbar.set_label('Log10 of |v_y|')

    # Set labels and title
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('v_y Velocity Field')
    plt.show()


# Function to plot concentration at a specific time
def plot_concentration_at_time(c_matrix, x, y, time_step, time_value):
    plt.figure()
    plt.imshow(np.log10(np.flipud(c_matrix[time_step, :, :])), extent=[x[0], x[-1], y[0], y[-1]], aspect='auto',
               cmap='jet')
    plt.colorbar(label='log10 Concentration')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Concentration at Time = {time_value:.2f} [Days]')
    # plt.axis('equal')
    plt.show()


# Function to plot multiple concentration plots for specified time steps
def plot_multiple_concentrations(c_matrix, x, y, time_steps):
    for time_step, time_value in time_steps:
        plot_concentration_at_time(c_matrix, x, y, time_step, time_value)


def plot_animation(c_matrix, x, y, T, K):
    # Time grid
    t = np.linspace(0, T, K + 1)
    X, Y = np.meshgrid(x, y)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    h_image = ax.imshow(np.flipud(np.log10(c_matrix[0, :, :])), extent=[x[0], x[-1], y[0], y[-1]], aspect='auto',
                        cmap='jet')
    plt.colorbar(h_image, ax=ax, label='log10 Concentration')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    title_text = ax.set_title('Time = 0 [days]')

    # Initialize the video writer
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    with writer.saving(fig, "ADE_with_pulse.mp4", dpi=100):
        for k in range(0, K + 1, 10):  # Save every 10th frame
            # Process the concentration matrix for finite values
            im_temp = np.full_like(c_matrix[k, :, :], np.nan)
            im_temp2 = np.log10(c_matrix[k, :, :])
            finite_mask = np.isfinite(im_temp2)
            im_temp[finite_mask] = im_temp2[finite_mask]

            # Update the image data
            h_image.set_data(np.flipud(im_temp))
            title_text.set_text(f'Time = {t[k]:.2f} [days]')

            # Write each frame to the video
            writer.grab_frame()

    plt.close(fig)


if __name__ == "__main__":
    # Defining Parameters

    # For ADE
    Lx = 160900  # Length of the spatial domain in x [cm]
    Ly = 199  # Length of the spatial domain in y [cm]
    T = 56  # Total time [min]
    D = 7 * 24 * 60 * 1E-9  # Molecular diffusion coefficient [m^2/days]
    Nx = 1609  # Number of spatial points in x [-]
    Ny = 199  # Number of spatial points in x [-]
    K = 200  # Number of time points [-]
    head_diff = 10  # Hydraulic head at inlet [m]
    c_inlet = 1  # Inlet concentration during the pulse [mg/cm^3]
    x = np.linspace(0, Lx, Nx + 1)  # Spatial grid in x (columns)
    y = np.linspace(0, Ly, Ny + 1)  # Spatial grid in y (rows)

    t_pulse_start = 0  # Start time of the pulse [days]
    t_pulse_end = 10  # End time of the pulse [days]

    [k_matrix, phi_matrix] = define_properties(Ny, Nx, 1)
    [h, v_x, v_y] = flow_solver(Lx, Ly, Nx, Ny, head_diff, k_matrix)

    num_snapshots = 10

    plot_velocity_field(x, y, v_x, v_y)
    plot_absolute_velocity(x, y, v_x, v_y)
    plot_vy_velocity_field(x, y, v_y)
    c_matrix = calc(Lx, Ly, T, v_x, v_y, D, Nx, Ny, K, num_snapshots)
