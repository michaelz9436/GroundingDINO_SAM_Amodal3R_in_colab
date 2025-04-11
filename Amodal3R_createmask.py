import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from Amodal3R.pipelines import Amodal3RImageTo3DPipeline
from Amodal3R.representations import Gaussian, MeshExtractResult
from Amodal3R.utils import render_utils, postprocessing_utils
from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
import cv2
import os
import shutil
import matplotlib.pyplot as plt


def reset_image(predictor, img):
    img = np.array(img)
    predictor.set_image(img)
    original_img = img.copy()
    return predictor, original_img, "The models are ready.", [], [], [], original_img


def run_sam(img, predictor, selected_points):
    if len(selected_points) == 0:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    input_points = [p for p in selected_points]
    input_labels = [1 for _ in range(len(selected_points))]
    masks, _, _ = predictor.predict(
        point_coords=np.array(input_points),
        point_labels=np.array(input_labels),
        multimask_output=False,
    )
    best_mask = masks[0].astype(np.uint8)
    # dilate
    if len(selected_points) > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best_mask = cv2.dilate(best_mask, kernel, iterations=1)
        best_mask = cv2.erode(best_mask, kernel, iterations=1)
    return best_mask


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


def unpack_state(state: dict) -> tuple:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh


def get_sam_predictor():
    sam_checkpoint = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def apply_mask_overlay(image, mask, color=(255, 0, 0)):
    img_arr = image
    overlay = img_arr.copy()
    gray_color = np.array([200, 200, 200], dtype=np.uint8)
    non_mask = mask == 0
    overlay[non_mask] = (0.5 * overlay[non_mask] + 0.5 * gray_color).astype(np.uint8)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def vis_mask(image, mask_list):
    updated_image = image.copy()
    combined_mask = np.zeros_like(updated_image[:, :, 0])
    for mask in mask_list:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    updated_image = apply_mask_overlay(updated_image, combined_mask)
    return updated_image


def segment_and_overlay(image, points, sam_predictor, mask_list, point_type):
    if point_type == "visibility":
        visible_mask = run_sam(image, sam_predictor, points)
        for mask in mask_list:
            visible_mask = cv2.bitwise_or(visible_mask, mask)
        overlaid = apply_mask_overlay(image, visible_mask * 255)
        return overlaid, visible_mask, mask_list
    else:
        combined_occlusion_mask = np.zeros_like(image[:, :, 0])
        mask_list = []
        if len(points) != 0:
            for point in points:
                mask = run_sam(image, sam_predictor, [point])
                mask_list.append(mask)
                combined_occlusion_mask = cv2.bitwise_or(combined_occlusion_mask, mask)
        overlaid = apply_mask_overlay(image, combined_occlusion_mask * 255, color=(0, 255, 0))
        return overlaid, combined_occlusion_mask, mask_list


def check_combined_mask(image, visibility_mask, visibility_mask_list, occlusion_mask_list, scale=0.68):
    if visibility_mask.sum() == 0:
        return np.zeros_like(image), np.zeros_like(image[:, :, 0])
    updated_image = image.copy()
    combined_mask = np.zeros_like(updated_image[:, :, 0])
    occluded_mask = np.zeros_like(updated_image[:, :, 0])
    binary_visibility_masks = [(m > 0).astype(np.uint8) for m in visibility_mask_list]
    combined_mask = np.zeros_like(binary_visibility_masks[0]) if binary_visibility_masks else (
                visibility_mask > 0).astype(np.uint8)
    for m in binary_visibility_masks:
        combined_mask = cv2.bitwise_or(combined_mask, m)

    if len(binary_visibility_masks) > 1:
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    binary_occlusion_masks = [(m > 0).astype(np.uint8) for m in occlusion_mask_list]
    occluded_mask = np.zeros_like(binary_occlusion_masks[0]) if binary_occlusion_masks else np.zeros_like(combined_mask)
    for m in binary_occlusion_masks:
        occluded_mask = cv2.bitwise_or(occluded_mask, m)

    kernel_small = np.ones((3, 3), np.uint8)
    if len(binary_occlusion_masks) > 0:
        dilated = cv2.dilate(combined_mask, kernel_small, iterations=1)
        boundary_mask = dilated - combined_mask
        occluded_mask = cv2.bitwise_or(occluded_mask, boundary_mask)
        occluded_mask = (occluded_mask > 0).astype(np.uint8)
        occluded_mask = cv2.dilate(occluded_mask, kernel_small, iterations=1)
        occluded_mask = (occluded_mask > 0).astype(np.uint8)
    else:
        occluded_mask = 1 - combined_mask

    combined_mask[occluded_mask == 1] = 0

    occluded_mask = (1 - occluded_mask) * 255

    masked_img = updated_image * combined_mask[:, :, None]
    occluded_mask[combined_mask == 1] = 127

    x, y, w, h = cv2.boundingRect(combined_mask.astype(np.uint8))

    ori_h, ori_w = masked_img.shape[:2]
    target_size = 512
    scale_factor = target_size / max(w, h)
    final_scale = scale_factor * scale
    new_w = int(round(ori_w * final_scale))
    new_h = int(round(ori_h * final_scale))

    resized_occluded_mask = cv2.resize(occluded_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized_img = cv2.resize(masked_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    final_img = np.zeros((target_size, target_size, 3), dtype=updated_image.dtype)
    final_occluded_mask = np.ones((target_size, target_size), dtype=np.uint8) * 255

    new_x = int(round(x * final_scale))
    new_y = int(round(y * final_scale))
    new_w_box = int(round(w * final_scale))
    new_h_box = int(round(h * final_scale))

    new_cx = new_x + new_w_box // 2
    new_cy = new_y + new_h_box // 2

    final_cx, final_cy = target_size // 2, target_size // 2
    x_offset = final_cx - new_cx
    y_offset = final_cy - new_cy

    final_x_start = max(0, x_offset)
    final_y_start = max(0, y_offset)
    final_x_end = min(target_size, x_offset + new_w)
    final_y_end = min(target_size, y_offset + new_h)

    img_x_start = max(0, -x_offset)
    img_y_start = max(0, -y_offset)
    img_x_end = min(new_w, target_size - x_offset)
    img_y_end = min(new_h, target_size - y_offset)

    final_img[final_y_start:final_y_end, final_x_start:final_x_end] = resized_img[img_y_start:img_y_end,
                                                                      img_x_start:img_x_end]
    final_occluded_mask[final_y_start:final_y_end, final_x_start:final_x_end] = resized_occluded_mask[
                                                                                img_y_start:img_y_end,
                                                                                img_x_start:img_x_end]

    return final_img, final_occluded_mask


def get_point(img, point_type, visible_points_state, occlusion_points_state, ind):
    if point_type == "visibility":
        visible_points_state = add_point(ind[0], ind[1], visible_points_state)
    else:
        occlusion_points_state = add_point(ind[0], ind[1], occlusion_points_state)
    return visible_points_state, occlusion_points_state


def add_point(x, y, visible_points):
    if [x, y] not in visible_points:
        visible_points.append([x, y])
    return visible_points


########################################################################################

predictor = get_sam_predictor()

visible_points =  [[760, 873], [808, 888], [818, 936], [804, 985], [794, 1014], [760, 1038], [726, 1024], [697, 990], [672, 946], [721, 941], [770, 941], [731, 873]]
occlusion_points = [[731, 907], [706, 888], [702, 917], [677, 917], [629, 932], [677, 864], [643, 893]]
visible_points_state = []
occlusion_points_state = []
occlusion_mask = None
original_image = None
visibility_mask = None
occluded_mask = None
visibility_mask_list = []
occlusion_mask_list = []

print("初始化成功，添加点")
input_image = "/content/Amodal3R/random_frame_3.jpg"
original_image = cv2.imread(input_image)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
predictor = reset_image(predictor, original_image)[0]

for ind in visible_points:
    visible_points_state, occlusion_points_state = get_point(original_image, "visibility", visible_points_state,occlusion_points_state, ind)
    render_mask, visibility_mask, visibility_mask_list = segment_and_overlay(original_image, visible_points_state, predictor, visibility_mask_list,"visibility")
    vis_input, occluded_mask = check_combined_mask(original_image, visibility_mask, visibility_mask_list, occlusion_mask_list, 0.68)


for ind in occlusion_points:
    visible_points_state, occlusion_points_state = get_point(original_image, "occlusion", visible_points_state,occlusion_points_state, ind)
    render_mask, occlusion_mask, occlusion_mask_list = segment_and_overlay(original_image, occlusion_points_state, predictor, occlusion_mask_list, "occlusion")
    vis_input, occluded_mask = check_combined_mask(original_image, visibility_mask, visibility_mask_list,occlusion_mask_list, 0.68)
print("sam完成，保存模型")
cv2.imwrite("/content/vis_input_cup.png", cv2.cvtColor(vis_input, cv2.COLOR_RGB2BGR))
cv2.imwrite("/content/occluded_mask_cup.png", occluded_mask)


plt.figure(figsize=(10, 5))

# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(vis_input)
plt.title('Final Image')
plt.axis('off')

# 遮挡图像
plt.subplot(1, 2, 2)
plt.imshow(occluded_mask, cmap='gray')
plt.title('Occluded Mask')
plt.axis('off')
# 保存到/content
plt.tight_layout()  # 自动调整布局，避免图像重叠
plt.savefig('/content/combined_output.png', bbox_inches='tight')

# 展示
plt.show()
###################################################################################3