# src/Work6/main.py
import taichi as ti
import math

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 10000.0      # 弹簧劲度系数
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

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
pull_force = ti.Vector.field(3, dtype=float, shape=()) # 計算出的滑鼠拉力向量
mouse_world_pos = ti.Vector.field(3, dtype=float, shape=()) # 滑鼠當前的 3D 位置

# ============ 初始化 ============

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0
    selected_vertex[None] = -1
    pull_force[None] = ti.Vector([0.0, 0.0, 0.0])

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
    """ 精確的 Z 軸近距離判斷：縮小半徑並嚴格尋找離射線最近的粒子 """
    min_dist = 999999.0
    target_idx = -1
    
    for i in range(N * N):
        p = x[i]
        v_vec = p - ray_origin
        projection = v_vec.dot(ray_dir)
        
        if projection > 0: # 確保在相機前方
            closest_point = ray_origin + projection * ray_dir
            dist = (p - closest_point).norm()
            
            # 【關鍵修改】：半徑縮小到 0.022 (因為粒子間距是 0.05，小於一半能防止點到隔壁)
            # 同時確保 dist < min_dist 嚴格挑選最接近射線中心的粒子
            if dist < 0.022 and dist < min_dist:
                min_dist = dist
                target_idx = i
                
    selected_vertex[None] = target_idx

@ti.kernel
def update_pull_force(mouse_x: float, mouse_y: float, start_x: float, start_y: float):
    """ 根據滑鼠在螢幕上拖曳的 2D 距離與方向，直接換算為 3D 世界空間的拉力 """
    idx = selected_vertex[None]
    if idx != -1:
        dx = mouse_x - start_x
        dy = mouse_y - start_y  # 往下拖動時 dy 為負值
        
        # 放大係數，讓拉力很有彈性
        k_pull = 50000.0  
        pull_force[None] = ti.Vector([dx * k_pull, dy * k_pull, 0.0])
        
        # 視覺化紅色橡皮筋終點位置
        mouse_world_pos[None] = x[idx] + ti.Vector([dx * 2.0, dy * 2.0, 0.0])

# ============ 力計算函數（加入滑鼠拉力） ============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template(), apply_pull: int):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
        
    if apply_pull == 1 and selected_vertex[None] != -1:
        force[selected_vertex[None]] += pull_force[None]
        
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

# ============ 積分 Kernel ============

@ti.kernel
def step_explicit(apply_pull: int):
    compute_forces_on(x, v, f, apply_pull)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)

@ti.kernel
def step_semi_implicit(apply_pull: int):
    compute_forces_on(x, v, f, apply_pull)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter(apply_pull: int):
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next, apply_pull)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

@ti.kernel
def reset_pull():
    selected_vertex[None] = -1
    pull_force[None] = ti.Vector([0.0, 0.0, 0.0])

# ============ 主函数 ============
def main():
    init_cloth()

    window = ti.ui.Window("Games101 - Mass Spring System (Interactive)", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    
    camera = ti.ui.Camera()
    
    # 1. 這裡我們固定住相機的物理數值，不再去讀取 camera 的內部成員
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
    apply_force_frames = 0  
    start_mouse_x = 0.0
    start_mouse_y = 0.0

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
        window.GUI.text("Left Click + Drag: Pull Vertex")
        window.GUI.text("Release Left Click: Release Force")
        window.GUI.text("Note: Camera is fixed for precise clicking")
        window.GUI.end()

        # ================== 完美的純數學 3D 射線互動處理 ==================
        mouse_x, mouse_y = window.get_cursor_pos()
        
        # 監聽滑鼠左鍵
        if window.is_pressed(ti.ui.LMB):
            if not is_dragging:
                start_mouse_x, start_mouse_y = mouse_x, mouse_y
                
                # 2. 將 2D 滑鼠位置用純數學轉換為 3D 射線 (100% 準確且不依賴 Taichi API)
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
                
                # 3. 呼叫穩定的 3D 射線求交 Kernel
                check_mouse_click_by_ray(ray_origin, ray_dir)
                
                if selected_vertex[None] != -1:
                    is_dragging = True
            else:
                # 正在拖拽中：根據起點差值更新拉力
                update_pull_force(mouse_x, mouse_y, start_mouse_x, start_mouse_y)
        else:
            if is_dragging:
                is_dragging = False
                apply_force_frames = 15  # 放開滑鼠時力道維持 15 個物理步，造成強力震動

        # ===================================================================

        if not paused:
            for _ in range(40):
                should_apply = 0
                if apply_force_frames > 0:
                    should_apply = 1
                    apply_force_frames -= 1
                    if apply_force_frames == 0:
                        reset_pull() 
                
                if current_method == 0:
                    step_explicit(should_apply)
                elif current_method == 1:
                    step_semi_implicit(should_apply)
                elif current_method == 2:
                    step_implicit_iter(should_apply)

        # 渲染场景 (拿掉了鏡頭旋轉功能，維持完美對齊的固定視角)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 绘制网格顶点和弹簧线框
        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        # 繪製拉力橡皮筋
        if is_dragging and selected_vertex[None] != -1:
            visual_line = ti.Vector.field(3, dtype=float, shape=2)
            visual_line[0] = x[selected_vertex[None]]
            visual_line[1] = mouse_world_pos[None]
            scene.lines(visual_line, width=4.0, color=(1.0, 0.2, 0.2))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()