# Grounding-DINO-SAM-Amodal-3R-in-colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/michaelz9436/GroundingDINO_SAM_Amodal3R_in_colab/blob/main/DINO_SAM_Amodal3R.ipynb)  
功能：输入图片和目标物体/遮挡物体的标签，采用Grounding sam生成mask，用Amodal3R对mask进行重建。另外还提供了nvdiffrast渲染mesh可视化  
[Open in colab] -> 在colab里按照指示运行，需要打开GPU运行时

几点注意：  
1.为了适配colab免费的t4，更改了Amodal3R中flash_attn为xformers，如果是flash_attn支持的A100（colab pro），可以根据文件内指引更改为原来的方法  
2.项目虽然用grounding dino，但仍保留了Amodal原来的sam方法（通过添加可视/遮挡点sam），可以根据文件内指引添加点生成mask 
3.Amodal3R支持的mask格式需要特殊处理，详见文件里处理mask的部分
4.由于colab运行Amodal3R原来的render方法显存爆炸，提供了nvdiffrast渲染mesh，但是你仍然可以下载打包的packed_state.pkl进行渲染  
5.可以使用内置的unpack_state方法将打包的state字典转化为Amodal3R支持的输出格式  
## 🔗 Based on

This project is based on the following open-source projects and demos:

- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)  
  A powerful combination of Grounding DINO and SAM, developed by IDEA Research, for open-set object detection and segmentation.

- [Amodal3R (Hugging Face Demo by Sm0kyWu)](https://huggingface.co/spaces/Sm0kyWu/Amodal3R)  
  A demo of Amodal 3D object segmentation and reconstruction using Amodal3R, shared by [Sm0kyWu](https://huggingface.co/Sm0kyWu).

All rights, licenses, and credits belong to the original authors. Please refer to their respective repositories and LICENSE files for more details.
