#!/usr/bin/env python  
#-*- coding:utf-8 _*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

idx = 0
start_idx = 6
path = '/root/files/pdessl/data/large/pdebench/ns3d_pdb_M1_rand/test/data_{}.hdf5'.format(idx)
data = h5py.File(path, 'r')['data'][...,start_idx, 2]
def volume_rendering(data, step=5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # 获取数据的尺寸
    nx, ny, nz = data.shape

    # 生成x, y, z坐标
    x, y, z = np.mgrid[0:nx:step, 0:ny:step, 0:nz:step]

    # 遍历每个点，绘制半透明的点
    for i in range(0, nx, step):
        for j in range(0, ny, step):
            for k in range(0, nz, step):
                ax.scatter(i, j, k, color='blue', alpha=data[i, j, k])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

x, y, z = np.mgrid[0:1:30j, 0:1:30j, 0:1:30j]
import plotly.graph_objects as go
# 创建 Plotly 图形对象
fig = go.Figure(data=go.Isosurface(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=data.flatten(),
    isomin=0.2,
    isomax=0.8,
    caps=dict(x_show=False, y_show=False),
    showscale=False
))

# 更新布局
# fig.update_layout()
fig.update_layout(
    scene=dict(
        xaxis_title='',  # 清空 X 轴标题
        yaxis_title='',  # 清空 Y 轴标题
        zaxis_title='',  # 清空 Z 轴标题
        xaxis_showticklabels=False,  # 隐藏 X 轴刻度标签
        yaxis_showticklabels=False,  # 隐藏 Y 轴刻度标签
        zaxis_showticklabels=False   # 隐藏 Z 轴刻度标签
    )
)
# fig.tight_layout()
# 显示图像
fig.show()

# 保存为静态图像
fig.write_image("pdb3d.png")


# volume_rendering(data)
# plot_iso_surface(data)