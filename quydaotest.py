import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

@st.cache_resource
def giai_ptvp():
    # Định nghĩa biến symbolic
    t, m, h, g, v0, alpha = sp.symbols('t m h g v0 alpha')
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)

    # Định nghĩa PTVP
    ode_x = sp.Eq(m * x.diff(t, 2), -h * x.diff(t, 1)) 
    ode_y = sp.Eq(m * y.diff(t, 2), -m*g - h * y.diff(t, 1)) 

    #Điều kiện ban đầu: x(0); y(0); v(0); 
    conds_x = {x.subs(t, 0): 0, x.diff(t, 1).subs(t, 0): v0*sp.cos(alpha)}
    conds_y = {y.subs(t, 0): 0, y.diff(t, 1).subs(t, 0): v0*sp.sin(alpha)}

    #Giải PTVP
    x_sol = sp.dsolve(ode_x, ics=conds_x)
    y_sol = sp.dsolve(ode_y, ics=conds_y)
    
    # Lấy nghiệm vận tốc
    vx_sol = sp.diff(x_sol.rhs, t)
    vy_sol = sp.diff(y_sol.rhs, t)
    
    #Dùng lambdify chuyển sang 4 hàm số numpy
    vars_tuple = (t, m, h, v0, alpha, g)
    x_func = sp.lambdify(vars_tuple, x_sol.rhs, 'numpy')
    y_func = sp.lambdify(vars_tuple, y_sol.rhs, 'numpy')
    vx_func = sp.lambdify(vars_tuple, vx_sol, 'numpy')
    vy_func = sp.lambdify(vars_tuple, vy_sol, 'numpy')
    
    return x_func, y_func, vx_func, vy_func

#Tính v = sqrt(v_x^2+v_y^2)
def tinh_v(val_x, val_y):
    return np.sqrt(val_x**2 + val_y**2)

st.set_page_config(layout="wide") 
st.title("Mô phỏng quỹ đạo chuyển động ném xiên trong trọng trường có lực cản môi trường")

x_func, y_func, vx_func, vy_func = giai_ptvp()

# Khởi tạo "Session State" để lưu trữ các quỹ đạo đã vẽ
if 'trajectories' not in st.session_state:
    st.session_state.trajectories = [] # List lưu các gói dữ liệu

# Danh sách các kiểu đường/màu sắc
styles = [
    {'color': 'blue', 'linestyle': '-'},
    {'color': 'red', 'linestyle': '--'},
    {'color': 'green', 'linestyle': ':'},
    {'color': 'orange', 'linestyle': '-.'},
    {'color': 'purple', 'linestyle': '-'},
    {'color': 'cyan', 'linestyle': '--'}
]

# Thanh điều khiển
st.sidebar.header("Thông số đầu vào")
alpha_deg = st.sidebar.slider("Góc bắn (độ):", 0, 90, 45)
v0_val = st.sidebar.slider("Vận tốc đầu (m/s):", 1, 200, 100)
h_val = st.sidebar.slider("Hệ số cản (h):", 0.01, 1.0, 0.1)
m_val = st.sidebar.slider("Khối lượng (m):", 0.1, 10.0, 1.0)
g_val = 9.81 



# 1. Tạo nút "Vẽ / Thêm"
if st.sidebar.button("Vẽ / Thêm Quỹ đạo này"):
    t_vec = np.linspace(0, 40, 1000) #Tạo 1000 giá trị t
    alpha_rad = np.deg2rad(alpha_deg)#Đổi độ sang rad
    
    # Tính các thành phần
    x_vals = x_func(t_vec, m_val, h_val, v0_val, alpha_rad, g_val)
    y_vals = y_func(t_vec, m_val, h_val, v0_val, alpha_rad, g_val)
    vx_vals = vx_func(t_vec, m_val, h_val, v0_val, alpha_rad, g_val)
    vy_vals = vy_func(t_vec, m_val, h_val, v0_val, alpha_rad, g_val)
    ax_vals = -(h_val / m_val) * vx_vals
    ay_vals = -g_val - (h_val / m_val) * vy_vals
    v_mag_vals = tinh_v(vx_vals, vy_vals)
    a_mag_vals = tinh_v(ax_vals, ay_vals)

    #Lọc những giá trị y<0
    indices = np.where(y_vals < 0) 
    end_index = indices[0][0] if indices[0].size > 0 else len(t_vec)
    
    # Tạo các gói dữ liệu
    data_pack = {
        "t": t_vec[:end_index],
        "x": x_vals[:end_index],
        "y": y_vals[:end_index],
        "v": v_mag_vals[:end_index],
        "a": a_mag_vals[:end_index],
        "label": f"α={alpha_deg}°, v₀={v0_val}, h={h_val}",
        "style": styles[len(st.session_state.trajectories) % len(styles)]
    }
    
    #Lưu vào session state
    st.session_state.trajectories.append(data_pack)

# Tạo nút "Xóa"
if st.sidebar.button("Xóa toàn bộ Đồ thị"):
    st.session_state.trajectories = []
    st.toast("Đã xóa đồ thị!")

# VẼ ĐỒ THỊ 
st.header("Đồ thị Vận tốc và Gia tốc (theo thời gian)")
col1, col2 = st.columns(2)

# Hàm vẽ lặp
def plot_multi_lines(ax, title, xlabel, ylabel, data_key, color_override=None):
    ax.set_title(title)
    if not st.session_state.trajectories:
        # vẽ khung trống
        pass
    else:
        # vẽ các đường đã lưu
        for traj in st.session_state.trajectories:
            style_color = traj["style"]["color"]
            ax.plot(
                traj["t"], traj[data_key], 
                label=traj["label"], 
                color=color_override if color_override else style_color, # Dùng màu riêng (cho a(t)) hoặc màu chung
                linestyle=traj["style"]["linestyle"], 
                linewidth=2
            )
        ax.legend(loc='best', fontsize='small')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

# Vận tốc và Gia tốc
with col1:
    fig_vel, ax_vel = plt.subplots()
    plot_multi_lines(ax_vel, "Độ lớn Vận tốc", "Thời gian (s)", "Vận tốc (m/s)", data_key="v")
    st.pyplot(fig_vel)

with col2:
    fig_accel, ax_accel = plt.subplots()
    plot_multi_lines(ax_accel, "Độ lớn Gia tốc", "Thời gian (s)", "Gia tốc (m/s²)", data_key="a")
    st.pyplot(fig_accel)

#Quỹ đạo
st.header("Đồ thị Quỹ đạo")
fig_traj, ax_traj = plt.subplots(figsize=(10, 7)) 

if st.session_state.trajectories:
    ax_traj.clear()
    ax_traj.set_title("Các quỹ đạo đã mô phỏng")
    for traj in st.session_state.trajectories:
        ax_traj.plot(
            traj["x"], traj["y"], 
            label=traj["label"], 
            color=traj["style"]["color"], 
            linestyle=traj["style"]["linestyle"], 
            linewidth=2
        )
    ax_traj.legend(loc='upper right', fontsize='small')

# lables các thứ
ax_traj.set_xlabel("Tầm xa x (m)")
ax_traj.set_ylabel("Độ cao y (m)")
ax_traj.grid(True)
ax_traj.set_ylim(bottom=0)
ax_traj.axis('equal') 
st.pyplot(fig_traj)