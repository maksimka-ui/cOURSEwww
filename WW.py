import tkinter as tk
from tkinter import ttk, messagebox
from solver_module import HeatEquationSolver
from visualization_module import visualize_2d_solution, visualize_3d_solution

class HeatSolverApp(tk.Tk):
    def __init__(self):
        """Инициализация главного окна приложения"""
        super().__init__()
        self.title("Решение задачи теплопроводности")
        self.geometry("610x610")
        self.create_widgets()

    def create_widgets(self):
        """Создание всех элементов интерфейса"""
        # Размерность задачи (2D или 3D)
        ttk.Label(self, text="Размерность задачи (2D или 3D):").pack(anchor='w', padx=10, pady=5)
        self.dim_var = tk.IntVar(value=2)
        ttk.Radiobutton(self, text="2D", variable=self.dim_var, value=2, command=self.on_dim_change).pack(anchor='w', padx=20)
        ttk.Radiobutton(self, text="3D", variable=self.dim_var, value=3, command=self.on_dim_change).pack(anchor='w', padx=20)

        # Контейнер для параметров сетки и области
        self.params_frame = ttk.Frame(self)
        self.params_frame.pack(fill='x', padx=10, pady=5)

        # Общие параметры с предустановленными значениями
        self.entries = {}
        params = [
            ("Количество узлов по x:", "nx", "10"),
            ("Количество узлов по y:", "ny", "10"),
            ("Длина области по x:", "Lx", "10"),
            ("Длина области по y:", "Ly", "10"),
            ("Начальная температура:", "T0", "10"),
            ("Шаг по времени dt:", "dt", "0.01"),
            ("Время моделирования:", "t_max", "10"),
        ]
        for text, key, default in params:
            frame = ttk.Frame(self.params_frame)
            frame.pack(fill='x', pady=3)
            ttk.Label(frame, text=text).pack(side='left')
            entry = ttk.Entry(frame)
            entry.pack(side='right', fill='x', expand=True)
            entry.insert(0, default)
            self.entries[key] = entry

        # Параметры для 3D — скрыты по умолчанию
        self.nz_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.nz_frame, text="Количество узлов по z:").pack(side='left')
        self.nz_entry = ttk.Entry(self.nz_frame)
        self.nz_entry.pack(side='right', fill='x', expand=True)
        self.nz_entry.insert(0, "10")

        self.Lz_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.Lz_frame, text="Длина области по z:").pack(side='left')
        self.Lz_entry = ttk.Entry(self.Lz_frame)
        self.Lz_entry.pack(side='right', fill='x', expand=True)
        self.Lz_entry.insert(0, "10")

        self.nz_frame.pack_forget()
        self.Lz_frame.pack_forget()

        # Граничные условия
        self.bc_frame = ttk.LabelFrame(self, text="Граничные условия (1-Дирихле, 2-Нейман, 3-Робина)")
        self.bc_frame.pack(fill='x', padx=10, pady=10)

        # Списки для переменных и виджетов
        self.bc_vars = []
        self.temp_entries = []  # Температура границы (для Дирихле и Неймана)
        self.env_temp_entries = []  # Температура окружающей среды (для Робина)

        # Изначально создаем для 2D
        self.create_bc_entries(dim=2)

        # Кнопка запуска расчетов
        ttk.Button(self, text="Запустить расчёт", command=self.run_solver).pack(pady=10)


    def create_bc_entries(self, dim):
        """Создание элементов интерфейса для граничных условий"""
        # Очищаем предыдущие виджеты в bc_frame
        for widget in self.bc_frame.winfo_children():
            widget.destroy()
        self.bc_vars.clear()
        self.temp_entries.clear()
        self.env_temp_entries.clear()

        # Порядок граней должен соответствовать порядку обработки в решателе:
        # Для 2D: ["левая", "правая", "нижняя", "верхняя"]
        # Для 3D: ["передняя", "задняя", "левая", "правая", "нижняя", "верхняя"]
        sides_2d = ["левая", "правая", "нижняя", "верхняя"]
        sides_3d = ["передняя", "задняя", "левая", "правая", "нижняя", "верхняя"]
        sides = sides_2d if dim == 2 else sides_3d

        for i, side in enumerate(sides):
            frame = ttk.Frame(self.bc_frame)
            frame.pack(fill='x', pady=4)

            ttk.Label(frame, text=f"Граничное условие на {side} грань:").pack(side='left')

            # Тип граничного условия
            var = tk.IntVar(value=1)
            self.bc_vars.append(var)
            combo = ttk.Combobox(frame, textvariable=var, values=[1, 2, 3], width=3, state='readonly')
            combo.pack(side='left', padx=5)
            combo.bind("<<ComboboxSelected>>", lambda e, idx=i: self.on_bc_type_change(idx))

            # Температура границы (Дирихле и Нейман)
            temp_frame = ttk.Frame(frame)
            ttk.Label(temp_frame, text="Темп. границы:").pack(side='left')
            temp_entry = ttk.Entry(temp_frame, width=8)
            temp_entry.insert(0, "10.0")
            temp_entry.pack(side='left')
            temp_frame.pack(side='left', padx=10)
            self.temp_entries.append(temp_frame)
            self.temp_entries.append(temp_entry)

            # Температура окружающей среды (Робин)
            env_frame = ttk.Frame(frame)
            ttk.Label(env_frame, text="Темп. окр. среды:").pack(side='left')
            env_entry = ttk.Entry(env_frame, width=8)
            env_entry.insert(0, "25.0")
            env_entry.pack(side='left')
            env_frame.pack_forget()
            self.env_temp_entries.append(env_frame)
            self.env_temp_entries.append(env_entry)

    def on_bc_type_change(self, idx):
        """Обработка изменения типа граничного условия"""
        bc_type = self.bc_vars[idx].get()
        temp_frame = self.temp_entries[idx * 2]
        temp_entry = self.temp_entries[idx * 2 + 1]
        env_frame = self.env_temp_entries[idx * 2]
        env_entry = self.env_temp_entries[idx * 2 + 1]

        if bc_type == 1:  # Дирихле
            temp_frame.pack(side='left', padx=10)
            env_frame.pack_forget()
        elif bc_type == 2:  # Нейман
            temp_frame.pack(side='left', padx=10)
            env_frame.pack_forget()
        elif bc_type == 3:  # Робин
            temp_frame.pack(side='left', padx=10)
            env_frame.pack(side='left', padx=10)

    def on_dim_change(self):
        """Обработка изменения размерности задачи"""
        dim = self.dim_var.get()
        if dim == 3:
            self.nz_frame.pack(fill='x', pady=3)
            self.Lz_frame.pack(fill='x', pady=3)
        else:
            self.nz_frame.pack_forget()
            self.Lz_frame.pack_forget()
        self.create_bc_entries(dim)

    def run_solver(self):
        """Запуск решения задачи теплопроводности"""
        try:
            dim = self.dim_var.get()
            nx = int(self.entries["nx"].get())
            ny = int(self.entries["ny"].get())
            nz = int(self.nz_entry.get()) if dim == 3 else 1
            Lx = float(self.entries["Lx"].get().replace(',', '.'))
            Ly = float(self.entries["Ly"].get().replace(',', '.'))
            Lz = float(self.Lz_entry.get().replace(',', '.')) if dim == 3 else 1.0
            T0 = float(self.entries["T0"].get().replace(',', '.'))
            dt = float(self.entries["dt"].get().replace(',', '.'))
            t_max = float(self.entries["t_max"].get().replace(',', '.'))

            bc_types = [var.get() for var in self.bc_vars]
            boundary_temps = []
            env_temps = []

            for i, bc in enumerate(bc_types):
                temp = float(self.temp_entries[i * 2 + 1].get())
                if bc == 3:
                    env_temp = float(self.env_temp_entries[i * 2 + 1].get())
                else:
                    env_temp = None
                boundary_temps.append(temp)
                env_temps.append(env_temp)

            solver = HeatEquationSolver(dim=dim, nx=nx, ny=ny, nz=nz,
                                     Lx=Lx, Ly=Ly, Lz=Lz, T0=T0,
                                     dt=dt, t_max=t_max,
                                     bc_types=bc_types,
                                     boundary_temps=boundary_temps,
                                     env_temps=env_temps)

            solution = solver.solve()

            if dim == 2:
                visualize_2d_solution(solution, Lx, Ly)
            else:
                visualize_3d_solution(solution, Lx, Ly, Lz)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка ввода или расчета:\n{e}")

if __name__ == "__main__":
    app = HeatSolverApp()
    app.mainloop()