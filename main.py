import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# =============================================================================
# КЛАСС: МАТЕМАТИЧЕСКИЙ ДВИЖОК (ЗАДАНИЕ 21)
# =============================================================================

class LSMSmoother:
    """
    Реализация сглаживания МНК: многочлен 2-й степени, 5 точек.
    Согласно методичке (стр. 25-26).
    """

    @staticmethod
    def process(Y: np.ndarray) -> tuple[np.ndarray, int]:
        N = len(Y)
        YY = np.copy(Y).astype(float)

        # Проверка минимального количества точек для 5-точечного шаблона
        if N < 5:
            return YY, 1

        # 1. Крайние точки Y[0] и Y[N-1] остаются без изменений (по методичке)

        # 2. Вторая точка (индекс 1) - формула (55)
        YY[1] = (18 * Y[0] + 26 * Y[1] + 24 * Y[2] + 12 * Y[3] - 10 * Y[4]) / 70.0

        # 3. Внутренние точки (от i=2 до N-3) - формула (55)
        for i in range(2, N - 2):
            YY[i] = (-3 * Y[i - 2] + 12 * Y[i - 1] + 17 * Y[i] + 12 * Y[i + 1] - 3 * Y[i + 2]) / 35.0

        # 4. Предпоследняя точка (индекс N-2) - формула (55)
        YY[N - 2] = (-10 * Y[N - 5] + 12 * Y[N - 4] + 24 * Y[N - 3] + 26 * Y[N - 2] + 18 * Y[N - 1]) / 70.0

        # Определение IER: 1 если значения совпали, 0 если изменились
        # Используем жесткий допуск, так как МНК для 2-й степени обязан
        # возвращать ту же параболу без искажений.
        is_same = np.allclose(Y, YY, atol=1e-9)
        ier = 1 if is_same else 0

        return YY, ier


# =============================================================================
# КЛАСС: ГЕНЕРАТОР ДАННЫХ И ОТЧЕТОВ
# =============================================================================

class LabWorkReport:
    def __init__(self):
        # Инициализируем генератор случайных чисел временем,
        # чтобы данные ВСЕГДА были разными при перезапуске
        np.random.seed(datetime.now().microsecond)

    def generate_random_case(self):
        """Генерирует случайную функцию с шумом."""
        n_points = np.random.randint(10, 25)
        x = np.linspace(0, 10, n_points)

        # Случайный выбор базовой функции
        base_type = np.random.choice(['sin', 'exp', 'poly'])
        if base_type == 'sin':
            y_pure = np.sin(x)
        elif base_type == 'exp':
            y_pure = np.exp(x / 10)
        else:
            y_pure = 0.1 * x ** 2 - 0.5 * x + 2

        # Добавляем случайный шум
        noise = np.random.normal(0, 0.15, n_points)
        return x, y_pure + noise, f"Случайная функция ({base_type}) с шумом"

    def print_detailed_table(self, Y, YY, ier):
        print("\n" + "═" * 85)
        print(f"{'ПОЛНЫЙ ТЕХНИЧЕСКИЙ ОТЧЕТ ПО СГЛАЖИВАНИЮ':^85}")
        print("═" * 85)
        print(f" Параметры: N = {len(Y)} | Метод: МНК (степень 2, 5 точек) | IER = {ier}")
        print("─" * 85)
        print(f"{'№':^5} | {'Исходное Y':^18} | {'Сглаженное YY':^18} | {'Коррекция (Δ)':^18}")
        print("─" * 85)

        for i in range(len(Y)):
            delta = YY[i] - Y[i]
            # Выделяем строки, где значения НЕ изменились (края)
            marker = " (край)" if i in [0, len(Y) - 1] else ""
            print(f"{i:^5} | {Y[i]:18.6f} | {YY[i]:18.6f} | {delta:18.6f}{marker}")

        print("─" * 85)
        rmse = np.sqrt(np.mean((YY - Y) ** 2))
        print(f" Среднеквадратичное отклонение (RMSE): {rmse:.8f}")
        print(f" Статус IER: {'[1] Данные инвариантны' if ier == 1 else '[0] Данные успешно сглажены'}")
        print("═" * 85)

    def plot_results(self, x, y, yy, title):
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8-muted')

        plt.plot(x, y, 'o--', color='gray', alpha=0.4, label='Исходный сигнал (Y)', markersize=5)
        plt.plot(x, yy, 'r-s', linewidth=2, label='Результат сглаживания (YY)', markersize=4)

        # Рисуем стрелки коррекции (ошибки)
        for i in range(len(x)):
            plt.vlines(x[i], y[i], yy[i], colors='blue', linestyles='dotted', alpha=0.5)

        plt.title(title, fontsize=14)
        plt.xlabel("Аргумент (равноотстоящие точки)")
        plt.ylabel("Значение функции")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


# =============================================================================
# ГЛАВНЫЙ ЦИКЛ
# =============================================================================

if __name__ == "__main__":
    report = LabWorkReport()

    # 1. СТРОГАЯ ПРОВЕРКА (Тест на параболе - Замечание 5)
    # Этот тест доказывает корректность реализации формул
    print("\n[ЭТАП 1] Математическая верификация (Тест на параболе)...")
    x_test = np.arange(8)
    y_test = 0.5 * x_test ** 2 - 2 * x_test + 1
    yy_test, ier_test = LSMSmoother.process(y_test)
    report.print_detailed_table(y_test, yy_test, ier_test)

    # 2. РАБОТА С НОВЫМИ ДАННЫМИ
    print("\n[ЭТАП 2] Генерация случайного набора данных...")
    x_rand, y_rand, desc = report.generate_random_case()
    yy_rand, ier_rand = LSMSmoother.process(y_rand)

    report.print_detailed_table(y_rand, yy_rand, ier_rand)
    report.plot_results(x_rand, y_rand, yy_rand, desc)