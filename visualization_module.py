import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_2d_solution(solution, Lx, Ly):
    """Визуализация решения для 2D случая"""
    plt.figure(figsize=(12, 5))

    # Контурный график
    plt.subplot(121)
    x = np.linspace(0, Lx, solution.shape[0])
    y = np.linspace(0, Ly, solution.shape[1])
    X, Y = np.meshgrid(x, y)

    contour = plt.contourf(X, Y, solution.T, levels=20, cmap='jet')
    plt.colorbar(label='Температура (°C)')
    plt.xlabel('Ось X (м)')
    plt.ylabel('Ось Y (м)')
    plt.title('Распределение температуры')

    # Графики вдоль осей
    plt.subplot(122)
    mid_x = solution.shape[0] // 2
    mid_y = solution.shape[1] // 2

    plt.plot(x, solution[:, mid_y], 'r-', label=f'При Y={y[mid_y]:.1f} м')
    plt.plot(y, solution[mid_x, :], 'b-', label=f'При X={x[mid_x]:.1f} м')

    plt.xlabel('Координата (м)')
    plt.ylabel('Температура (°C)')
    plt.title('Температура вдоль осей')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def visualize_3d_solution(solution, Lx, Ly, Lz):
    """Визуализация решения для 3D случая"""
    fig = plt.figure(figsize=(15, 5))

    # Срезы по осям
    slices = [
        ('Срез по X', solution[solution.shape[0] // 2, :, :], Ly, Lz, 'Y', 'Z'),
        ('Срез по Y', solution[:, solution.shape[1] // 2, :], Lx, Lz, 'X', 'Z'),
        ('Срез по Z', solution[:, :, solution.shape[2] // 2], Lx, Ly, 'X', 'Y')
    ]

    for i, (title, slc, dim1, dim2, xlabel, ylabel) in enumerate(slices, 1):
        ax = fig.add_subplot(1, 3, i)
        contour = ax.contourf(np.linspace(0, dim1, slc.shape[0]),
                            np.linspace(0, dim2, slc.shape[1]),
                            slc.T, 20, cmap='jet')
        plt.colorbar(contour, ax=ax, label='Температура (°C)')
        ax.set_xlabel(f'{xlabel} (м)')
        ax.set_ylabel(f'{ylabel} (м)')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()