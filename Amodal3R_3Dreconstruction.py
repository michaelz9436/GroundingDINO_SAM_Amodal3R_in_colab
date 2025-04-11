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

    # 查看 Gaussian 对象的属性和方法
    #print(outputs)
    gaussian_obj = outputs['gaussian'][0]  # 获取 Gaussian 对象
    #print(gaussian_obj.__dict__)  # 输出所有属性
    #video = render_utils.render_video(outputs['gaussian'][0], num_frames=120, bg_color=(1, 1, 1))['color']
    #video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    #video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]

    # 保存渲染视频
    #video_path = os.path.join(output_dir, 'sample.mp4')
    #imageio.mimsave(video_path, video, fps=15)

    # 打包返回的状态数据
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])

    # 清空 GPU 缓存
    torch.cuda.empty_cache()

    return state


output_dir = '/content/3d_output'  # 你希望保存文件的目录
image = cv2.imread('/content/vis_input.png')# 读取输入图像和掩码
mask = cv2.imread('/content/occluded_mask.png', cv2.IMREAD_GRAYSCALE)

# 运行 3D 生成
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

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Gaussian Point Cloud Visualization")
    plt.savefig("point_cloud.png")  # 保存图像
    from IPython.display import display
    from PIL import Image
    display(Image.open("point_cloud.png"))  # 在 Colab 里显示
    
    plt.show()
show_point_cloud_matplotlib(state)
print("点云输出")

def show_mesh_matplotlib(state):
    vertices = state['mesh']['vertices']  # 提取顶点坐标
    faces = state['mesh']['faces']  # 提取面数据
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制网格
    for face in faces:
        # face是一个包含3个顶点的三角形面
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        x = [v1[0], v2[0], v3[0], v1[0]]
        y = [v1[1], v2[1], v3[1], v1[1]]
        z = [v1[2], v2[2], v3[2], v1[2]]
        ax.plot_trisurf(x, y, z, color='cyan', linewidth=0.1, alpha=0.5)

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Mesh Visualization")
    plt.savefig("mesh.png")  # 保存网格图像
    from IPython.display import display
    from PIL import Image
    display(Image.open("mesh.png"))  # 在 Colab 里显示

    plt.show()

#show_mesh_matplotlib(state)



