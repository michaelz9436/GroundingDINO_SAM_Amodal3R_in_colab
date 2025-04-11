import os
import numpy as np
import cv2
import pickle
import torch
import imageio
from Amodal3R.pipelines import Amodal3RImageTo3DPipeline
from Amodal3R.utils import render_utils
from Amodal3R.representations import Gaussian, MeshExtractResult


# 加载 3D 生成模型
pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
pipeline.cuda()

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
def image_to_3d(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 50,
    slat_guidance_strength: float = 7.5,
    slat_sampling_steps: int = 50,
    erode_kernel_size: int = 5,
    output_dir: str = '/content/3d_output'
):
    """生成3D点云和网格，并保存到文件"""
    os.makedirs(output_dir, exist_ok=True)

    print("输入图像尺寸:", image.shape)
    print("输入掩码尺寸:", mask.shape)

    # 运行 Amodal3R 生成 3D 结构
    outputs = pipeline.run_multi_image(
        [image],
        [mask],
        seed=seed,
        formats=["gaussian", "mesh"],  # 生成高斯点云和网格
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
        mode="stochastic",
        erode_kernel_size=erode_kernel_size,
    )


    # 打包返回的状态数据
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])

    # 清空 GPU 缓存
    torch.cuda.empty_cache()

    return state


output_dir = '/content/3d_output'  #保存文件目录
image = cv2.imread('/content/finalreal.png')  # 读取输入图像和掩码
mask = cv2.imread('/content/finalmask.png', cv2.IMREAD_GRAYSCALE)
"""
运行 3D 生成
在此更改相关生成参数
"""
state = image_to_3d(
    image=image,
    mask=mask,
    seed=42,
    ss_guidance_strength=7.5,
    ss_sampling_steps=12,
    slat_guidance_strength=3,
    slat_sampling_steps=12,
    erode_kernel_size=3,
    output_dir='/content/3d_output'
)


###########   打包   ############   打包   ############
with open('/content/packed_state.pkl', 'wb') as f:
    pickle.dump(state, f)
###########   打包   ############   打包   ############

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print("点云数量",len(state['gaussian']['_xyz']))
print("mesh顶点数量",len(state['mesh']['vertices']))


def show_point_cloud_matplotlib(state):
    points = state['gaussian']['_xyz']  # 提取点云坐标
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画出点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', marker='o', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Gaussian Point Cloud Visualization")
    plt.savefig("point_cloud.png")  # 保存图像
    from IPython.display import display
    from PIL import Image
    display(Image.open("point_cloud.png")) 
    
    plt.show()
show_point_cloud_matplotlib(state)
print("点云输出，请查看/content/point_cloud.png")
