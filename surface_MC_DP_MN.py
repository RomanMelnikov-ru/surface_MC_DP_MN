import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

# Начальные параметры материала
c_initial = 10  # Начальное значение сцепления (кПа)
phi_initial = 20  # Начальное значение угла внутреннего трения (градусы)
plane_constant_initial = 120  # Начальное значение для плоскости

# Универсальная функция для вычисления sigma3 или sigma2 через sigma1 (или наоборот)
def edge(sigma, c, phi, mode):
    if mode == "12>3":
        return sigma - (2 * c * np.cos(phi) + 2 * sigma * np.sin(phi)) / (1 + np.sin(phi))
    elif mode == "12<3":
        return sigma + (2 * c * np.cos(phi) + 2 * sigma * np.sin(phi)) / (1 - np.sin(phi))
    elif mode == "13>2":
        return sigma - (2 * c * np.cos(phi) + 2 * sigma * np.sin(phi)) / (1 + np.sin(phi))
    elif mode == "13<2":
        return sigma + (2 * c * np.cos(phi) + 2 * sigma * np.sin(phi)) / (1 - np.sin(phi))
    elif mode == "23>1":
        return sigma - (2 * c * np.cos(phi) + 2 * sigma * np.sin(phi)) / (1 + np.sin(phi))
    elif mode == "23<1":
        return sigma + (2 * c * np.cos(phi) + 2 * sigma * np.sin(phi)) / (1 - np.sin(phi))

# Функция для нахождения пересечения ребра с плоскостью
def find_intersection(edge_func, sigma_range, plane_constant, vertex, axis, c, phi, mode):
    sigma_values = sigma_range
    if axis == 1:  # sigma1 = sigma2 > sigma3 или sigma1 = sigma2 < sigma3
        sigma3 = edge_func(sigma_values, c, phi, mode)
        mask = sigma_values + sigma_values + sigma3 >= plane_constant
        if np.any(mask):
            idx = np.argmax(mask)
            return np.array([sigma_values[idx], sigma_values[idx], sigma3[idx]])
    elif axis == 2:  # sigma1 = sigma3 > sigma2 или sigma1 = sigma3 < sigma2
        sigma2 = edge_func(sigma_values, c, phi, mode)
        mask = sigma_values + sigma2 + sigma_values >= plane_constant
        if np.any(mask):
            idx = np.argmax(mask)
            return np.array([sigma_values[idx], sigma2[idx], sigma_values[idx]])
    elif axis == 3:  # sigma2 = sigma3 > sigma1 или sigma2 = sigma3 < sigma1
        sigma1 = edge_func(sigma_values, c, phi, mode)
        mask = sigma1 + sigma_values + sigma_values >= plane_constant
        if np.any(mask):
            idx = np.argmax(mask)
            return np.array([sigma1[idx], sigma_values[idx], sigma_values[idx]])
    return None

# Функция для обновления графика
def update(c, phi, plane_constant, camera_state=None):
    phi_rad = np.radians(phi)
    sigma_vertex = -c / np.tan(phi_rad)
    vertex = np.array([sigma_vertex, sigma_vertex, sigma_vertex])
    intersections = [
        find_intersection(edge, np.linspace(sigma_vertex, 100, 1000), plane_constant, vertex, 1, c, phi_rad, "12>3"),
        find_intersection(edge, np.linspace(sigma_vertex, 100, 1000), plane_constant, vertex, 2, c, phi_rad, "13>2"),
        find_intersection(edge, np.linspace(sigma_vertex, 100, 1000), plane_constant, vertex, 3, c, phi_rad, "23>1"),
        find_intersection(edge, np.linspace(sigma_vertex, 100, 1000), plane_constant, vertex, 1, c, phi_rad, "12<3"),
        find_intersection(edge, np.linspace(sigma_vertex, 100, 1000), plane_constant, vertex, 2, c, phi_rad, "13<2"),
        find_intersection(edge, np.linspace(sigma_vertex, 100, 1000), plane_constant, vertex, 3, c, phi_rad, "23<1")
    ]
    fig = go.Figure()

    # Группа для критерия Мора-Кулона
    mora_coulomb_group = "Поверхность Мора-Кулона"

    # Рисуем ребра (Мора-Кулона)
    colors = ['red', 'red', 'red', 'red', 'red', 'red']
    for i, (intersection, color) in enumerate(zip(intersections, colors)):
        if intersection is not None:
            fig.add_trace(go.Scatter3d(
                x=[vertex[0], intersection[0]],
                y=[vertex[1], intersection[1]],
                z=[vertex[2], intersection[2]],
                mode='lines',
                line=dict(color=color, width=1),
                legendgroup=mora_coulomb_group,
                name=mora_coulomb_group,
                showlegend=False  # Показываем легенду только для первого элемента
            ))

    # Рисуем вершину (Мора-Кулона)
    fig.add_trace(go.Scatter3d(
        x=[vertex[0]],
        y=[vertex[1]],
        z=[vertex[2]],
        mode='markers',
        marker=dict(size=3, color='red'),
        legendgroup=mora_coulomb_group,
        name=mora_coulomb_group,
        showlegend=False
    ))

    # Рисуем грани пирамиды (Мора-Кулона)
    if all(intersection is not None for intersection in intersections):
        all_points = [vertex] + intersections
        x = [point[0] for point in all_points]
        y = [point[1] for point in all_points]
        z = [point[2] for point in all_points]
        # Индексы для граней
        i = [0, 0, 0, 0, 0, 0]  # Вершина пирамиды
        j = [5, 1, 6, 2, 4, 3]  # Первая точка грани
        k = [1, 6, 2, 4, 3, 5]  # Вторая точка грани
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='red',
            opacity=0.2,
            legendgroup=mora_coulomb_group,
            name=mora_coulomb_group,
            showlegend=True
        ))

    # Группа для критерия Друкера-Прагера
    drucker_prager_group = "Поверхность Друкера-Прагера"

    # Добавляем окружность и конус Друкера-Прагера
    alpha = (2 * np.sin(phi_rad)) / (np.sqrt(3) * (3 - np.sin(phi_rad)))
    k = (6 * c * np.cos(phi_rad)) / (np.sqrt(3) * (3 - np.sin(phi_rad)))
    radius = np.sqrt(2) * (k + alpha * plane_constant)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    sigma1_circle = plane_constant / 3 + (x_circle / np.sqrt(2)) - (y_circle / np.sqrt(6))
    sigma2_circle = plane_constant / 3 - (x_circle / np.sqrt(2)) - (y_circle / np.sqrt(6))
    sigma3_circle = plane_constant / 3 + (2 * y_circle / np.sqrt(6))

    # Вершина конуса Друкера-Прагера
    sigma_vertex_dp = -np.array([k / (3 * alpha), k / (3 * alpha), k / (3 * alpha)])

    # Поверхность конуса Друкера-Прагера
    u = np.linspace(0, 1, 100)  # Параметр для интерполяции
    sigma1_cone = np.outer(1 - u, sigma1_circle) + np.outer(u, sigma_vertex_dp[0])
    sigma2_cone = np.outer(1 - u, sigma2_circle) + np.outer(u, sigma_vertex_dp[1])
    sigma3_cone = np.outer(1 - u, sigma3_circle) + np.outer(u, sigma_vertex_dp[2])

    # Окружность (критерий Друкера-Прагера)
    fig.add_trace(go.Scatter3d(
        x=sigma1_circle,
        y=sigma2_circle,
        z=sigma3_circle,
        mode='lines',
        line=dict(color='green', width=2),
        legendgroup=drucker_prager_group,
        name=drucker_prager_group,
        showlegend=False
    ))

    # Поверхность конуса Друкера-Прагера
    fig.add_trace(go.Surface(
        x=sigma1_cone,
        y=sigma2_cone,
        z=sigma3_cone,
        colorscale='Greens',
        opacity=0.5,
        showscale=False,
        legendgroup=drucker_prager_group,
        name=drucker_prager_group,
        showlegend=True
    ))

    # Вершина конуса Друкера-Прагера
    fig.add_trace(go.Scatter3d(
        x=[sigma_vertex_dp[0]],
        y=[sigma_vertex_dp[1]],
        z=[sigma_vertex_dp[2]],
        mode='markers',
        marker=dict(color='green', size=3),
        legendgroup=drucker_prager_group,
        name=drucker_prager_group,
        showlegend=False
    ))

    # Группа для критерия Матсуока-Накай
    matsouka_nakai_group = "Поверхность Матсуока-Накай"

    # Вычисление точек для критерия Матсуока-Накай
    tan_phi = np.tan(phi_rad)
    const = 9 + 8 * tan_phi**2
    sigma_range = np.linspace(-300, 300, 700)  # Увеличенный диапазон и количество точек
    sigma_1, sigma_2 = np.meshgrid(sigma_range, sigma_range)
    sigma_3 = plane_constant - sigma_1 - sigma_2

    # Эффективные напряжения (учет сцепления)
    sigma_1_eff = sigma_1 + c / tan_phi
    sigma_2_eff = sigma_2 + c / tan_phi
    sigma_3_eff = sigma_3 + c / tan_phi

    # Вычисление инвариантов
    I1 = sigma_1_eff + sigma_2_eff + sigma_3_eff
    I2 = sigma_1_eff * sigma_2_eff + sigma_2_eff * sigma_3_eff + sigma_3_eff * sigma_1_eff
    I3 = sigma_1_eff * sigma_2_eff * sigma_3_eff + 1e-6  # Добавляем небольшое значение, чтобы избежать деления на ноль

    # Уравнение критерия Матсуока-Накай
    criterion = (I1 * I2) / I3

    # Поиск точек, где выполняется условие разрушения
    surface_points = np.abs(criterion - const) < 0.02  # Уточненный порог для визуализации

    # Фильтрация точек вне пирамиды
    pyramid_mask = (sigma_1_eff >= 0) & (sigma_2_eff >= 0) & (sigma_3_eff >= 0)
    surface_points = surface_points & pyramid_mask

    # Извлечение координат точек поверхности
    sigma_1_surface = sigma_1[surface_points]
    sigma_2_surface = sigma_2[surface_points]
    sigma_3_surface = sigma_3[surface_points]

    # Вершина критерия Матсуока-Накай
    sigma_vertex_mn = np.array([sigma_vertex, sigma_vertex, sigma_vertex])

    # Соединение точек линиями (используем ConvexHull для построения контура)
    if len(sigma_1_surface) > 3:  # Минимум 4 точки для построения выпуклой оболочки
        points = np.vstack((sigma_1_surface, sigma_2_surface, sigma_3_surface)).T
        hull = ConvexHull(points[:, :2])  # Используем только sigma_1 и sigma_2 для 2D выпуклой оболочки

        # След в девиаторной плоскости (Матсуока-Накай)
        for simplex in hull.simplices:
            fig.add_trace(go.Scatter3d(
                x=points[simplex, 0],
                y=points[simplex, 1],
                z=points[simplex, 2],
                mode='lines',
                line=dict(color='blue', width=1),
                legendgroup=matsouka_nakai_group,
                name=matsouka_nakai_group,
                showlegend=False  # Показываем легенду только для первого элемента
            ))

        # Создание поверхности для критерия Матсуока-Накай
        x_mn = np.concatenate([points[hull.vertices, 0], [sigma_vertex_mn[0]]])
        y_mn = np.concatenate([points[hull.vertices, 1], [sigma_vertex_mn[1]]])
        z_mn = np.concatenate([points[hull.vertices, 2], [sigma_vertex_mn[2]]])

        # Индексы для поверхности
        i_mn = np.arange(len(hull.vertices))
        j_mn = np.roll(i_mn, -1)
        k_mn = np.full_like(i_mn, len(hull.vertices))  # Вершина

        fig.add_trace(go.Mesh3d(
            x=x_mn,
            y=y_mn,
            z=z_mn,
            i=i_mn,
            j=j_mn,
            k=k_mn,
            color='blue',
            opacity=0.2,
            legendgroup=matsouka_nakai_group,
            name=matsouka_nakai_group,
            showlegend=True
        ))

    # Вершина критерия Матсуока-Накай
    fig.add_trace(go.Scatter3d(
        x=[sigma_vertex_mn[0]],
        y=[sigma_vertex_mn[1]],
        z=[sigma_vertex_mn[2]],
        mode='markers',
        marker=dict(color='blue', size=3),
        legendgroup=matsouka_nakai_group,
        name=matsouka_nakai_group,
        showlegend=False
    ))

    # След в девиаторной плоскости для Мора-Кулона
    if all(intersection is not None for intersection in intersections):
        # Для каждой из 6 линий явно указываем порядок соединения точек
        line_order = [
            (5, 1), (1, 3), (3, 2), (2, 4), (4, 0), (0, 5)
        ]
        for start, end in line_order:
            start_point = intersections[start]
            end_point = intersections[end]
            fig.add_trace(go.Scatter3d(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                z=[start_point[2], end_point[2]],
                mode='lines',
                line=dict(color='red', width=1),
                legendgroup=mora_coulomb_group,
                name=mora_coulomb_group,
                showlegend=False
            ))

    # Гидростатическая ось
    hydrostatic_line = np.linspace(sigma_vertex, sigma_vertex + 100, 50)
    fig.add_trace(go.Scatter3d(
        x=hydrostatic_line,
        y=hydrostatic_line,
        z=hydrostatic_line,
        mode='lines',
        line=dict(color='grey', width=2, dash='dash'),
        name='Гидростатическая ось',
        showlegend=False
    ))

    # Оси начала координат
    fig.add_trace(go.Scatter3d(
        x=[0, 50], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='red', width=3),
        name='Ось σ₁',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 50], z=[0, 0],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Ось σ₂',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, 50],
        mode='lines',
        line=dict(color='green', width=3),
        name='Ось σ₃',
        showlegend=False
    ))

    # Настройки графика
    fig.update_layout(
        title='Поверхности критериев Мора-Кулона, Друкера-Прагера, Матсуока-Накай',
        scene=dict(
            xaxis_title='σ₂',
            yaxis_title='σ₃',
            zaxis_title='σ₁',
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            tracegroupgap=5,
            itemsizing='constant'
        )
    )

    # Сохраняем текущее состояние камеры
    if camera_state:
        fig.update_layout(scene_camera=camera_state)
    return fig

# Создаем слайдеры в Streamlit
c = st.slider('Удельное сцепление (c, кПа)', 0, 40, c_initial, key='c_slider')
phi = st.slider('Угол внутреннего трения (fi, °)', 10, 30, phi_initial, key='phi_slider')
plane_constant = st.slider('Положение девиаторной плоскости (σ₁ + σ₂ + σ₃)', 0, 200, plane_constant_initial, key='plane_slider')

# Поле результатов для параметров Друкера-Прагера и Матсуока-Накай
alpha = (2 * np.sin(np.radians(phi))) / (np.sqrt(3) * (3 - np.sin(np.radians(phi))))
k = (6 * c * np.cos(np.radians(phi))) / (np.sqrt(3) * (3 - np.sin(np.radians(phi))))
st.write(f"Параметры Друкера-Прагера: α = {alpha:.4f}, k = {k:.4f}")

const = 9 + 8 * np.tan(np.radians(phi))**2
st.write(f"Параметр Матсуока-Накай: const = {const:.4f}")

# Используем session state для сохранения состояния камеры
if 'camera_state' not in st.session_state:
    st.session_state['camera_state'] = dict(eye=dict(x=1.5, y=1.5, z=1.5))

# Обновляем и отображаем график
fig = update(c, phi, plane_constant, camera_state=st.session_state['camera_state'])
st.plotly_chart(fig)

# Сохраняем новое состояние камеры при взаимодействии с графиком
st.session_state['camera_state'] = fig.layout.scene.camera
