import pygame
import numpy as np

# ---------------------------
# Simulation Parameters
# ---------------------------
N = 128          # Simulation grid resolution
size = N + 2     # +2 for boundary cells
dt = 0.05        # Time step
diff = 0.0001    # Diffusion rate
visc = 0.0001    # Viscosity

# Fluid arrays: current and previous states
u       = np.zeros((size, size))  # x-velocity
v       = np.zeros((size, size))  # y-velocity
u_prev  = np.zeros((size, size))
v_prev  = np.zeros((size, size))

dens       = np.zeros((size, size))  # density
dens_prev  = np.zeros((size, size))

# ---------------------------
# Core Fluid Functions
# ---------------------------
def add_source(x, s, dt):
    x += dt * s

def set_bnd(b, x):
    """
    Sets boundary conditions.
    b == 1: horizontal velocity
    b == 2: vertical velocity
    b == 0: density or pressure
    """
    x[0, 1:-1]   = -x[1, 1:-1]   if b == 1 else x[1, 1:-1]
    x[-1, 1:-1]  = -x[-2, 1:-1]  if b == 1 else x[-2, 1:-1]
    x[1:-1, 0]   = -x[1:-1, 1]   if b == 2 else x[1:-1, 1]
    x[1:-1, -1]  = -x[1:-1, -2]  if b == 2 else x[1:-1, -2]

    # corners
    x[0, 0]       = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1]      = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0]      = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1]     = 0.5 * (x[-2, -1] + x[-1, -2])

def diffuse(b, x, x0, diff, dt):
    a = dt * diff * N * N
    for _ in range(20):  # Gauss-Seidel iterations
        x[1:-1, 1:-1] = (
            x0[1:-1, 1:-1] + a * (
                x[0:-2, 1:-1] +
                x[2:,   1:-1] +
                x[1:-1, 0:-2] +
                x[1:-1, 2:]
            )
        ) / (1 + 4*a)
        set_bnd(b, x)

def advect(b, d, d0, u, v, dt):
    dt0 = dt * N
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]

            if x < 0.5: x = 0.5
            if x > N + 0.5: x = N + 0.5
            if y < 0.5: y = 0.5
            if y > N + 0.5: y = N + 0.5

            i0, j0 = int(x), int(y)
            i1, j1 = i0 + 1, j0 + 1

            s1, s0 = x - i0, 1 - (x - i0)
            t1, t0 = y - j0, 1 - (y - j0)

            d[i, j] = (
                s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
            )
    set_bnd(b, d)

def project(u, v, p, div):
    h = 1.0 / N

    div[1:-1, 1:-1] = -0.5 * h * (
        u[2:, 1:-1] - u[0:-2, 1:-1] +
        v[1:-1, 2:] - v[1:-1, 0:-2]
    )
    p.fill(0)
    set_bnd(0, div)
    set_bnd(0, p)

    for _ in range(20):
        p[1:-1, 1:-1] = (
            div[1:-1, 1:-1] +
            p[0:-2, 1:-1] +
            p[2:,   1:-1] +
            p[1:-1, 0:-2] +
            p[1:-1, 2:]
        ) / 4.0
        set_bnd(0, p)

    u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[0:-2, 1:-1]) / h
    v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, 0:-2]) / h
    set_bnd(1, u)
    set_bnd(2, v)

def vel_step(u, v, u_prev, v_prev, visc, dt):
    add_source(u, u_prev, dt)
    add_source(v, v_prev, dt)

    # Diffuse
    u, u_prev = u_prev, u
    diffuse(1, u, u_prev, visc, dt)

    v, v_prev = v_prev, v
    diffuse(2, v, v_prev, visc, dt)

    # Project
    project(u, v, u_prev, v_prev)

    # Advect
    u, u_prev = u_prev, u
    v, v_prev = v_prev, v
    advect(1, u, u_prev, u_prev, v_prev, dt)
    advect(2, v, v_prev, u_prev, v_prev, dt)

    # Project again
    project(u, v, u_prev, v_prev)

def dens_step(x, x0, u, v, diff, dt):
    add_source(x, x0, dt)

    x, x0 = x0, x
    diffuse(0, x, x0, diff, dt)

    x, x0 = x0, x
    advect(0, x, x0, u, v, dt)

# ---------------------------
# Pygame Setup
# ---------------------------
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2D Fluid Simulation - By Shaun Harris")
clock = pygame.time.Clock()

running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Mouse input
    mouse_buttons = pygame.mouse.get_pressed()
    mouse_x, mouse_y = pygame.mouse.get_pos()
    i = int((mouse_x / width) * N + 1)
    j = int((mouse_y / height) * N + 1)

    if mouse_buttons[0]:

        if 1 <= i <= N and 1 <= j <= N:
            dens_prev[i, j] += 5_000_000 * dt  # Density source
            rel_x, rel_y = pygame.mouse.get_rel()
            u_prev[i, j] += rel_x * dt * 50
            v_prev[i, j] += rel_y * dt * 50
    else:
        # Reset mouse relative if not pressed
        pygame.mouse.get_rel()

    # Update fluid
    vel_step(u, v, u_prev, v_prev, visc, dt)
    dens_step(dens, dens_prev, u, v, diff, dt)

    # Clear the accumulators
    u_prev.fill(0)
    v_prev.fill(0)
    dens_prev.fill(0)

    # ---------------------------
    # Rendering to a small surface
    # ---------------------------
    # 1) Create a surface the size of N x N
    fluid_surface = pygame.Surface((N, N))

    # 2) Fill each pixel based on density
    #    (We skip boundary cells in dens)
    for grid_x in range(N):
        for grid_y in range(N):
            d = dens[grid_x + 1, grid_y + 1] * 5
            d = max(0, min(255, int(d)))
            color = (d, d, d)
            fluid_surface.set_at((grid_x, grid_y), color)

    # 3) Scale it to the window size
    scaled_surface = pygame.transform.scale(fluid_surface, (width, height))

    # 4) Blit the scaled surface
    screen.blit(scaled_surface, (0, 0))

    # Flip the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
