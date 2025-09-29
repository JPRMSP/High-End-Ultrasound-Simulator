import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="High-End Ultrasound Simulator", layout="wide")
st.title("🩺 High-End Ultrasound Simulator")
st.write("3D Ultrasound with B-mode, Doppler, Elastography, and Flow Streamlines Simulation.")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
freq = st.sidebar.slider("Ultrasound Frequency (MHz)", 1, 15, 5)
amp = st.sidebar.slider("Pulse Amplitude", 0.1, 1.0, 0.5)
tissue_speed = st.sidebar.slider("Sound Speed in Tissue (m/s)", 1400, 1600, 1540)
doppler_velocity = st.sidebar.slider("Blood Flow Velocity (m/s)", 0.0, 2.0, 0.5)
scatter_density = st.sidebar.slider("Scatterer Density", 100, 1000, 500)
simulation_speed = st.sidebar.slider("Simulation Speed (s)", 0.01, 0.2, 0.05)
stiffness_variation = st.sidebar.slider("Tissue Stiffness Variation", 0.0, 1.0, 0.5)

# 3D tissue grid
depth = 40
width = 40
elevation = 30

# Generate scatterers
scatterers = np.random.rand(scatter_density, 3)
scatterers[:,0] *= width
scatterers[:,1] *= depth
scatterers[:,2] *= elevation

# Streamlit placeholders
st.subheader("3D Ultrasound Volume")
ultrasound_3d_plot = st.empty()

st.subheader("Scan Plane Control")
scan_plane = st.slider("Scan Plane (Elevation Slice)", 0, elevation-1, 0)

st.subheader("2D B-mode Slice")
bmode_plot = st.empty()

st.subheader("Doppler Waveform")
doppler_plot = st.empty()

st.subheader("Elastography Simulation")
elasto_plot = st.empty()

# Simulation loop
for i in range(150):
    # Move scatterers along depth
    scatterers[:,1] += doppler_velocity * simulation_speed * 5
    scatterers[:,1] %= depth
    
    # 3D Doppler flow: simple streamline particles
    x = scatterers[:,0]
    y = scatterers[:,1]
    z = scatterers[:,2]
    color = np.full_like(x, doppler_velocity)
    
    fig3d = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=color, colorscale='RdBu', cmin=0, cmax=2, opacity=0.8)
    )])
    
    fig3d.update_layout(scene=dict(
        xaxis_title='Lateral (cm)',
        yaxis_title='Depth (cm)',
        zaxis_title='Elevation (cm)',
        xaxis=dict(range=[0,width]),
        yaxis=dict(range=[0,depth]),
        zaxis=dict(range=[0,elevation])
    ),
    margin=dict(l=0,r=0,b=0,t=30),
    title=f"3D Ultrasound Volume - Scan Plane {scan_plane+1}/{elevation}")
    
    ultrasound_3d_plot.plotly_chart(fig3d, use_container_width=True)
    
    # 2D B-mode slice
    mask = (scatterers[:,2] >= scan_plane) & (scatterers[:,2] < scan_plane + 1)
    plane_scatterers = scatterers[mask]
    bmode_img = np.zeros((depth, width))
    for s in plane_scatterers:
        xi, yi = int(s[0]), int(s[1])
        if 0 <= xi < width and 0 <= yi < depth:
            bmode_img[yi, xi] += 1.0
    bmode_img = bmode_img / np.max(bmode_img) if np.max(bmode_img) > 0 else bmode_img
    plt.figure(figsize=(8,4))
    plt.imshow(bmode_img, cmap='gray', aspect='auto', extent=[0,5,5,0])
    plt.title("2D B-mode Slice")
    plt.xlabel("Lateral (cm)")
    plt.ylabel("Depth (cm)")
    bmode_plot.pyplot(plt.gcf())
    plt.close()
    
    # Doppler waveform
    t_axis = np.linspace(0, 0.05, 500)
    doppler_signal = amp * np.sin(2*np.pi*freq*1e6*(1 + doppler_velocity/tissue_speed)*t_axis)
    plt.figure(figsize=(8,2))
    plt.plot(t_axis*1000, doppler_signal, color='red')
    plt.title("Simulated Doppler Waveform")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.ylim(-1,1)
    plt.grid(True)
    doppler_plot.pyplot(plt.gcf())
    plt.close()
    
    # Elastography simulation: simulate tissue stiffness as displacement-based color map
    elasto_img = np.zeros((depth, width))
    displacement = (np.sin(2*np.pi*(x+y+i*0.1)/width)*stiffness_variation)**2
    for idx, s in enumerate(plane_scatterers):
        xi, yi = int(s[0]), int(s[1])
        if 0 <= xi < width and 0 <= yi < depth:
            elasto_img[yi, xi] = displacement[idx]
    elasto_img = elasto_img / np.max(elasto_img) if np.max(elasto_img) > 0 else elasto_img
    plt.figure(figsize=(8,4))
    plt.imshow(elasto_img, cmap='viridis', aspect='auto', extent=[0,5,5,0])
    plt.title("Simulated Elastography (Tissue Stiffness)")
    plt.xlabel("Lateral (cm)")
    plt.ylabel("Depth (cm)")
    elasto_plot.pyplot(plt.gcf())
    plt.close()
    
    time.sleep(simulation_speed)

st.success("✅ High-End Ultrasound Simulation Complete! Explore 3D volume, B-mode, Doppler, elastography, and scan planes interactively.")
