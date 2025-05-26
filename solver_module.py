import numpy as np


class HeatEquationSolver:
    def __init__(self, dim, nx, ny, nz, Lx, Ly, Lz, T0, dt, t_max, bc_types, boundary_temps, env_temps):
        """
        Инициализация решателя уравнения теплопроводности
        Параметры:
        dim - размерность задачи (2 или 3)
        nx, ny, nz - количество узлов по осям
        Lx, Ly, Lz - размеры области
        T0 - начальная температура
        dt - шаг по времени
        t_max - время моделирования
        bc_types - типы граничных условий для каждой грани
        boundary_temps - температуры границ или градиенты
        env_temps - температуры окружающей среды для условий Робина
        """
        self.dim = dim
        self.nx = nx
        self.ny = ny
        self.nz = nz if dim == 3 else 1
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz if dim == 3 else 1.0
        self.T0 = T0
        self.dt = dt
        self.t_max = t_max
        self.bc_types = bc_types
        self.boundary_temps = boundary_temps
        self.env_temps = env_temps

        # Расчет шагов сетки
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dz = Lz / (nz - 1) if dim == 3 else 0

        # Коэффициент температуропроводности (можно сделать параметром)
        self.alpha = 1.0

    def solve(self):
        """Основной метод для решения задачи"""
        if self.dim == 2:
            return self._solve_2d()
        else:
            return self._solve_3d()

    def _solve_2d(self):
        """Решение двумерной задачи методом расщепления"""
        # Инициализация массива решения
        u = np.full((self.nx, self.ny), self.T0, dtype=float)

        # Количество шагов по времени
        nt = int(self.t_max / self.dt)

        # Основной цикл по времени
        for _ in range(nt):
            # Первый шаг: решение в направлении x (неявная схема)
            u = self._solve_x_implicit_2d(u)

            # Второй шаг: решение в направлении y (неявная схема)
            u = self._solve_y_implicit_2d(u)

        return u

    def _solve_3d(self):
        """Решение трехмерной задачи методом расщепления"""
        # Инициализация массива решения
        u = np.full((self.nx, self.ny, self.nz), self.T0, dtype=float)

        # Количество шагов по времени
        nt = int(self.t_max / self.dt)

        # Основной цикл по времени
        for _ in range(nt):
            # Схема расщепления для 3D
            u = self._solve_x_implicit_3d(u)
            u = self._solve_y_implicit_3d(u)
            u = self._solve_z_implicit_3d(u)

        return u

    def _solve_x_implicit_2d(self, u):
        """Решение одномерной задачи в направлении x для 2D случая"""
        u_new = u.copy()
        r = self.alpha * self.dt / (self.dx ** 2)

        # Решение для каждого среза по y
        for j in range(1, self.ny - 1):
            main_diag = np.full(self.nx, 1 + 2 * r)
            rhs = u[:, j].copy()

            # Левая граница (x=0)
            bc_type = self.bc_types[0]  # левая граница
            if bc_type == 1:  # Дирихле
                main_diag[0] = 1
                rhs[0] = self.boundary_temps[0]
            elif bc_type == 2:  # Нейман
                main_diag[0] = 1 + r
                rhs[0] = u[0, j] + r * self.dx * self.boundary_temps[0]
            elif bc_type == 3:  # Робин
                h = self.boundary_temps[0]  # коэффициент теплоотдачи
                T_env = self.env_temps[0]
                main_diag[0] = 1 + r + r * h * self.dx
                rhs[0] = u[0, j] + r * h * self.dx * T_env

            # Правая граница (x=Lx)
            bc_type = self.bc_types[1]  # правая граница
            if bc_type == 1:  # Дирихле
                main_diag[-1] = 1
                rhs[-1] = self.boundary_temps[1]
            elif bc_type == 2:  # Нейман
                main_diag[-1] = 1 + r
                rhs[-1] = u[-1, j] - r * self.dx * self.boundary_temps[1]
            elif bc_type == 3:  # Робин
                h = self.boundary_temps[1]
                T_env = self.env_temps[1]
                main_diag[-1] = 1 + r + r * h * self.dx
                rhs[-1] = u[-1, j] + r * h * self.dx * T_env

            # Заполнение диагоналей и решение СЛАУ
            lower_diag = np.full(self.nx - 1, -r)
            upper_diag = np.full(self.nx - 1, -r)
            u_new[:, j] = self._solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        return u_new

    def _solve_y_implicit_2d(self, u):
        """Решение одномерной задачи в направлении y для 2D случая"""
        u_new = u.copy()
        r = self.alpha * self.dt / (self.dy ** 2)

        # Решение для каждого среза по x
        for i in range(1, self.nx - 1):
            main_diag = np.full(self.ny, 1 + 2 * r)
            rhs = u[i, :].copy()

            # Нижняя граница (y=0)
            bc_type = self.bc_types[2]  # нижняя граница
            if bc_type == 1:  # Дирихле
                main_diag[0] = 1
                rhs[0] = self.boundary_temps[2]
            elif bc_type == 2:  # Нейман
                main_diag[0] = 1 + r
                rhs[0] = u[i, 0] + r * self.dy * self.boundary_temps[2]
            elif bc_type == 3:  # Робин
                h = self.boundary_temps[2]
                T_env = self.env_temps[2]
                main_diag[0] = 1 + r + r * h * self.dy
                rhs[0] = u[i, 0] + r * h * self.dy * T_env

            # Верхняя граница (y=Ly)
            bc_type = self.bc_types[3]  # верхняя граница
            if bc_type == 1:  # Дирихле
                main_diag[-1] = 1
                rhs[-1] = self.boundary_temps[3]
            elif bc_type == 2:  # Нейман
                main_diag[-1] = 1 + r
                rhs[-1] = u[i, -1] - r * self.dy * self.boundary_temps[3]
            elif bc_type == 3:  # Робин
                h = self.boundary_temps[3]
                T_env = self.env_temps[3]
                main_diag[-1] = 1 + r + r * h * self.dy
                rhs[-1] = u[i, -1] + r * h * self.dy * T_env

            lower_diag = np.full(self.ny - 1, -r)
            upper_diag = np.full(self.ny - 1, -r)
            u_new[i, :] = self._solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        return u_new

    def _solve_x_implicit_3d(self, u):
        """Решение одномерной задачи в направлении x для 3D случая"""
        u_new = u.copy()
        r = self.alpha * self.dt / (self.dx ** 2)

        for j in range(1, self.ny - 1):
            for k in range(1, self.nz - 1):
                main_diag = np.full(self.nx, 1 + 2 * r)
                rhs = u[:, j, k].copy()

                # Левая граница (x=0)
                bc_type = self.bc_types[2]  # левая граница (индекс 2 в 3D)
                if bc_type == 1:  # Дирихле
                    main_diag[0] = 1
                    rhs[0] = self.boundary_temps[2]
                elif bc_type == 2:  # Нейман
                    main_diag[0] = 1 + r
                    rhs[0] = u[0, j, k] + r * self.dx * self.boundary_temps[2]
                elif bc_type == 3:  # Робин
                    h = self.boundary_temps[2]
                    T_env = self.env_temps[2]
                    main_diag[0] = 1 + r + r * h * self.dx
                    rhs[0] = u[0, j, k] + r * h * self.dx * T_env

                # Правая граница (x=Lx)
                bc_type = self.bc_types[3]  # правая граница (индекс 3 в 3D)
                if bc_type == 1:  # Дирихле
                    main_diag[-1] = 1
                    rhs[-1] = self.boundary_temps[3]
                elif bc_type == 2:  # Нейман
                    main_diag[-1] = 1 + r
                    rhs[-1] = u[-1, j, k] - r * self.dx * self.boundary_temps[3]
                elif bc_type == 3:  # Робин
                    h = self.boundary_temps[3]
                    T_env = self.env_temps[3]
                    main_diag[-1] = 1 + r + r * h * self.dx
                    rhs[-1] = u[-1, j, k] + r * h * self.dx * T_env

                lower_diag = np.full(self.nx - 1, -r)
                upper_diag = np.full(self.nx - 1, -r)
                u_new[:, j, k] = self._solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        return u_new

    def _solve_y_implicit_3d(self, u):
        """Решение одномерной задачи в направлении y для 3D случая"""
        u_new = u.copy()
        r = self.alpha * self.dt / (self.dy ** 2)

        for i in range(1, self.nx - 1):
            for k in range(1, self.nz - 1):
                main_diag = np.full(self.ny, 1 + 2 * r)
                rhs = u[i, :, k].copy()

                # Нижняя граница (y=0)
                bc_type = self.bc_types[4]  # нижняя граница (индекс 4 в 3D)
                if bc_type == 1:  # Дирихле
                    main_diag[0] = 1
                    rhs[0] = self.boundary_temps[4]
                elif bc_type == 2:  # Нейман
                    main_diag[0] = 1 + r
                    rhs[0] = u[i, 0, k] + r * self.dy * self.boundary_temps[4]
                elif bc_type == 3:  # Робин
                    h = self.boundary_temps[4]
                    T_env = self.env_temps[4]
                    main_diag[0] = 1 + r + r * h * self.dy
                    rhs[0] = u[i, 0, k] + r * h * self.dy * T_env

                # Верхняя граница (y=Ly)
                bc_type = self.bc_types[5]  # верхняя граница (индекс 5 в 3D)
                if bc_type == 1:  # Дирихле
                    main_diag[-1] = 1
                    rhs[-1] = self.boundary_temps[5]
                elif bc_type == 2:  # Нейман
                    main_diag[-1] = 1 + r
                    rhs[-1] = u[i, -1, k] - r * self.dy * self.boundary_temps[5]
                elif bc_type == 3:  # Робин
                    h = self.boundary_temps[5]
                    T_env = self.env_temps[5]
                    main_diag[-1] = 1 + r + r * h * self.dy
                    rhs[-1] = u[i, -1, k] + r * h * self.dy * T_env

                lower_diag = np.full(self.ny - 1, -r)
                upper_diag = np.full(self.ny - 1, -r)
                u_new[i, :, k] = self._solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        return u_new

    def _solve_z_implicit_3d(self, u):
        """Решение одномерной задачи в направлении z для 3D случая"""
        u_new = u.copy()
        r = self.alpha * self.dt / (self.dz ** 2)

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                main_diag = np.full(self.nz, 1 + 2 * r)
                rhs = u[i, j, :].copy()

                # Передняя граница (z=0)
                bc_type = self.bc_types[0]  # передняя граница (индекс 0 в 3D)
                if bc_type == 1:  # Дирихле
                    main_diag[0] = 1
                    rhs[0] = self.boundary_temps[0]
                elif bc_type == 2:  # Нейман
                    main_diag[0] = 1 + r
                    rhs[0] = u[i, j, 0] + r * self.dz * self.boundary_temps[0]
                elif bc_type == 3:  # Робин
                    h = self.boundary_temps[0]
                    T_env = self.env_temps[0]
                    main_diag[0] = 1 + r + r * h * self.dz
                    rhs[0] = u[i, j, 0] + r * h * self.dz * T_env

                # Задняя граница (z=Lz)
                bc_type = self.bc_types[1]  # задняя граница (индекс 1 в 3D)
                if bc_type == 1:  # Дирихле
                    main_diag[-1] = 1
                    rhs[-1] = self.boundary_temps[1]
                elif bc_type == 2:  # Нейман
                    main_diag[-1] = 1 + r
                    rhs[-1] = u[i, j, -1] - r * self.dz * self.boundary_temps[1]
                elif bc_type == 3:  # Робин
                    h = self.boundary_temps[1]
                    T_env = self.env_temps[1]
                    main_diag[-1] = 1 + r + r * h * self.dz
                    rhs[-1] = u[i, j, -1] + r * h * self.dz * T_env

                lower_diag = np.full(self.nz - 1, -r)
                upper_diag = np.full(self.nz - 1, -r)
                u_new[i, j, :] = self._solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        return u_new

    def _solve_tridiagonal(self, a, b, c, d):
        """
        Решение трехдиагональной системы уравнений методом прогонки (Томаса)

        Параметры:
        a - нижняя диагональ (длина n-1)
        b - главная диагональ (длина n)
        c - верхняя диагональ (длина n-1)
        d - правая часть (длина n)

        Возвращает:
        x - решение системы
        """
        n = len(d)
        cp = np.zeros(n - 1)
        dp = np.zeros(n)

        # Прямой ход прогонки
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]

        for i in range(1, n - 1):
            denom = b[i] - a[i - 1] * cp[i - 1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

        dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-1])

        # Обратный ход прогонки
        x = np.zeros(n)
        x[-1] = dp[-1]

        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]

        return x