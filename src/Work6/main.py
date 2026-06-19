# src/Work6/main.py
import taichi as ti
import math

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 20000.0      # 提高弹簧劲度系数，让形变拉扯感更紧绷、震动更明显
k_d = 1.5          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点 (0:自由, 1:永久固定, 2:滑鼠暫時固定)

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧数据场
max_springs = N * N * 4
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

# ============ 滑鼠互動相關的 Taichi 資料場 ============
selected_vertex = ti.field(dtype=int, shape=())       # 當前選中的質點 ID (-1 表示沒選中)

# ============ 初始化 ============

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1  # 1 代表左上、右上角永久固定
        else:
            is_fixed[idx] = 0  # 0 代表自由點
    selected_vertex[None] = -1

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

@ti.kernel
def check_mouse_click_by_ray(ray_origin: ti.types.vector(3, float), ray_dir: ti.types.vector(3, float)):
    """ 精確尋找滑鼠點擊的粒子，並將其狀態設為暫時固定 """
    min_dist = 999999.0
    target_idx = -1
    
    for i in range(N * N):
        p = x[i]
        v_vec = p - ray_origin
        projection = v_vec.dot(ray_dir)
        
        if projection > 0:
            closest_point = ray_origin + projection * ray_dir
            dist = (p - closest_point).norm()
            if dist < 0.022 and dist < min_dist:
                min_dist = dist
                target_idx = i
                
    if target_idx != -1:
        selected_vertex[None] = target_idx
        # 如果不是原本那兩個角落的永久固定點(1)，就將其設為滑鼠固定狀態(2)
        if is_fixed[target_idx] == 0:
            is_fixed[target_idx] = 2

@ti.kernel
def update_dragged_vertex_position(mouse_x: float, mouse_y: float, start_x: float, start_y: float, start_pos: ti.types.vector(3, float)):
    """ 根據滑鼠在螢幕上的拖曳量，即時更新被選中粒子的 3D 空間位置 """
    idx = selected_vertex[None]
    if idx != -1:
        dx = mouse_x - start_x
        dy = mouse_y - start_y
        
        # 將螢幕 2D 拖曳量轉為 3D 位移（乘以 1.5 放大係數讓移動手感更同步）
        # 粒子會老老實實跟著滑鼠走，進而拉扯整張布料產生形變
        x[idx] = start_pos + ti.Vector([dx * 1.5, dy * 1.5, 0.0])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])  # 被拖曳時速度歸零

@ti.kernel
def release_vertex():
    """ 釋放滑鼠，將質點回復為自由狀態，讓積蓄的形變彈性能釋放 """
    idx = selected_vertex[None]
    if idx != -1:
        if is_fixed[idx] == 2:
            is_fixed[idx] = 0  # 回復為自由點
    selected_vertex[None] = -1

# ============ 力計算函數 (純物理，不施加外力，靠形變連動) ============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
        
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -k_s * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

# ============ 積分 Kernel (只更新自由點 i.e. is_fixed == 0) ============

@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:  # 只有自由點會受物理力更新，固定點（包含滑鼠抓取的點）不動
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] = v_next[i]
            x[i] = x_next[i]

# ============ 主函数 ============
def main():
    init_cloth()

    window = ti.ui.Window("Cloth Real-time Deformation (Interactive)", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    
    camera = ti.ui.Camera()
    
    cam_pos = ti.Vector([0.0, 0.5, 2.0])
    cam_lookat = ti.Vector([0.0, 0.0, 0.0])
    cam_up = ti.Vector([0.0, 1.0, 0.0])
    fov = 45.0
    aspect_ratio = 1.0

    camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
    camera.lookat(cam_lookat[0], cam_lookat[1], cam_lookat[2])

    current_method = 1 
    paused = False
    
    is_dragging = False
    start_mouse_x = 0.0
    start_mouse_y = 0.0
    vertex_start_pos = ti.Vector([0.0, 0.0, 0.0]) # 記錄點擊時粒子的初始 3D 位置

    while window.running:
        # =========== GUI 控制面板 ===========
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.36)
        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")
        if window.GUI.button("Resume Simulation" if paused else "Pause Simulation"):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()
            
        window.GUI.text("\n[Instructions]")
        window.GUI.text("Left Click + Drag: Pull Cloth directly")
        window.GUI.text("Release Left Click: Snap back & Vibrate")
        window.GUI.text("Note: Camera is fixed for precise clicking")
        window.GUI.end()

        # ================== 即時形變互動處理 ==================
        mouse_x, mouse_y = window.get_cursor_pos()
        
        if window.is_pressed(ti.ui.LMB):
            if not is_dragging:
                start_mouse_x, start_mouse_y = mouse_x, mouse_y
                
                # 1. 射線求交計算
                nx = mouse_x * 2.0 - 1.0
                ny = mouse_y * 2.0 - 1.0
                cam_forward = (cam_lookat - cam_pos).normalized()
                cam_right = cam_forward.cross(cam_up).normalized()
                cam_actual_up = cam_right.cross(cam_forward).normalized()
                tan_half_fov = math.tan(math.radians(fov / 2.0))
                world_x = nx * aspect_ratio * tan_half_fov
                world_y = ny * tan_half_fov
                ray_origin = cam_pos
                ray_dir = (cam_forward + world_x * cam_right + world_y * cam_actual_up).normalized()
                
                check_mouse_click_by_ray(ray_origin, ray_dir)
                
                # 2. 如果選中點，記錄點擊當下的 3D 位置作為拖曳基準
                if selected_vertex[None] != -1:
                    is_dragging = True
                    # 從 Taichi field 把被選中點的當前位置讀出來
                    idx = selected_vertex[None]
                    vertex_start_pos = ti.Vector([x[idx][0], x[idx][1], x[idx][2]])
            else:
                # 3. 正在拖曳中：即時把滑鼠位移量加到粒子位置上
                update_dragged_vertex_position(mouse_x, mouse_y, start_mouse_x, start_mouse_y, vertex_start_pos)
        else:
            if is_dragging:
                # 4. 放開滑鼠：粒子解鎖，能量釋放
                release_vertex()
                is_dragging = False

        # ===================================================================

        # 就算滑鼠按著不動，物理模擬依然在運作！
        # 這會讓其他粒子即時向滑鼠拉扯的位置下垂、緊繃
        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # 渲染场景
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 調整粒子渲染大小（radius=0.025），更容易看清和精確點選
        scene.particles(x, radius=0.020, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()