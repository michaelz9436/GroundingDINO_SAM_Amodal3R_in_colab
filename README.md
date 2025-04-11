# Grounding-DINO-SAM-Amodal-3R-in-colab
下载ipynb文件，放进colab里跟着指示运行，需要GPU运行时

几点修改：
1.使用了https://github.com/IDEA-Research/Grounded-Segment-Anything和https://huggingface.co/spaces/Sm0kyWu/Amodal3R提供的方法
2.为了适配colab免费的t4，更改了Amodal3R中flash_attn为xformers，如果是A100，可以根据文件内指引不做更改
3.项目虽然用grounding dino，仍保留了Amodal原来的sam方法（通过添加可视/遮挡点sam），可以根据文件内指引尝试
4.由于colab运行Amodal3R原来的render方法显存爆炸，提供了nvdiffrast渲染mesh，但是你仍然可以下载打包的packed_state.pkl进行渲染
5.部分地方仍待优化
