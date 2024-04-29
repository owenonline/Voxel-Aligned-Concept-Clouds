"""
This is the script I wrote to take the output from the step 1 notebook and run ConceptFusion to 
generate features which I could compare to the ones generated with VACC.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm, trange
import glob
import argparse
from tqdm import tqdm
import open_clip
from PIL import Image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages
from typing_extensions import Literal
import json
import math
from time import time

from lavis.models.eva_vit import create_eva_vit_g

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512

def create_masks(save_dir, outputs):
    torch.autograd.set_grad_enabled(False)

    # load the SAM model
    sam = sam_model_registry["vit_h"](checkpoint=Path("sam_vit_h_4b8939.pth"))
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    mask_time = 0

    for img_name in ['sceneshotcam', 'apple', 'milk', 'cereal', 'bread', 'banana', 'bin', 'ur5e', 'panda', 'table']:
        mask_start = time()
        savefile = os.path.join(
            save_dir, os.path.basename(img_name).replace(".png", ".pt")
        )
        if os.path.exists(savefile):
            continue
        
        # read in the picture
        imgfile = img_name
        img = outputs[img_name][0]

        # generate the masks for all the objects in the image
        masks = mask_generator.generate(img)
        _savefile = os.path.join(
                save_dir,
                os.path.splitext(os.path.basename(imgfile))[0] + ".pt",
            )
        
        # stack the masks and save them as a tensor
        mask_list = []
        for mask_item in masks:
            mask_list.append(mask_item["segmentation"])
        mask_np = np.asarray(mask_list)
        mask_torch = torch.from_numpy(mask_np)

        mask_end = time()
        mask_time += mask_end - mask_start
        
        torch.save(mask_torch, _savefile)

    print(f"mask time: {mask_time}")

def get_bbox_around_mask(mask):
    # mask: (img_height, img_width)
    # compute bbox around mask
    bbox = None
    nonzero_inds = torch.nonzero(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft = nonzero_inds.min(0)[0]  # (2,)
        botright = nonzero_inds.max(0)[0]  # (2,)
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds


def clip_sam(save_dir_path, mask_dir_path, outputs):
    OPENCLIP_MODEL = "ViT-L-14"  # "ViT-bigG-14"
    OPENCLIP_DATA = "laion2b_s32b_b82k"  # "laion2b_s39b_b160k"
    model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL, OPENCLIP_DATA)
    model.visual.output_tokens = True
    model.cuda()
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)

    clip_time = 0

    for file in ['sceneshotcam', 'apple', 'milk', 'cereal', 'bread', 'banana', 'bin', 'ur5e', 'panda', 'table']:
        clip_start = time()
        SEMIGLOBAL_FEAT_SAVE_FILE = os.path.join(save_dir_path, f"{file}_rgb.pt")
        raw_image = outputs[file][0]
        image = torch.tensor(raw_image)

        """
        Extract and save global feat vec
        """
        global_feat = None
        # with torch.cuda.amp.autocast():
        _img = preprocess(Image.fromarray(outputs[file][0])).unsqueeze(0).cuda()  # [1, 3, 224, 224]
        imgfeat = model.visual(_img)[1]  # All image token feat [1, 256, 1024]
        imgfeat = torch.mean(imgfeat, dim=1)

        global_feat = imgfeat.half().cuda()

        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        FEAT_DIM = global_feat.shape[-1]

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        MASK_LOAD_FILE = os.path.join(mask_dir_path, f"{file}.pt")
        outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, FEAT_DIM, dtype=torch.half).cuda()

        mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)  # 1, num_masks, H, W
        num_masks = mask.shape[-3]

        rois = []
        roi_similarities_with_global_vec = []
        roi_sim_per_unit_area = []
        feat_per_roi = []
        roi_nonzero_inds = []

        for _i in range(num_masks):
            curmask = mask[0, _i]
            bbox, nonzero_inds = get_bbox_around_mask(curmask)
            x0, y0, x1, y1 = bbox

            bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
            iou = bbox_area / img_area

            if iou < 0.005:
                continue
            with torch.no_grad():
                img_roi = image[x0:x1, y0:y1]
                img_roi = Image.fromarray(img_roi.numpy())
                img_roi = preprocess(img_roi).unsqueeze(0).cuda()
                roifeat = model.visual(img_roi)[1]  # All image token feat [1, 256, 1024]
                roifeat = torch.mean(roifeat, dim=1)

                feat_per_roi.append(roifeat)
                roi_nonzero_inds.append(nonzero_inds)
                _sim = cosine_similarity(global_feat, roifeat)

                rois.append(torch.tensor(list(bbox)))
                roi_similarities_with_global_vec.append(_sim)
                roi_sim_per_unit_area.append(_sim)

        rois = torch.stack(rois).cuda()
        scores = torch.cat(roi_sim_per_unit_area).to('cuda')
        retained = torchvision.ops.nms(rois.float().cuda(), scores.float().cuda(), iou_threshold=1.0).cuda()
        feat_per_roi = torch.cat(feat_per_roi, dim=0).cuda()  # N, 1024

        retained_rois = rois[retained].cuda()
        retained_scores = scores[retained].cuda()
        retained_feat = feat_per_roi[retained].cuda()
        retained_nonzero_inds = []
        for _roiidx in range(retained.shape[0]):
            retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])

        mask_sim_mat = torch.nn.functional.cosine_similarity(
            retained_feat[:, :, None], retained_feat.t()[None, :, :]
        ).cuda()
        mask_sim_mat.fill_diagonal_(0.0)
        mask_sim_mat = mask_sim_mat.mean(1).cuda()  # avg sim of each mask with each other mask
        softmax_scores = retained_scores - mask_sim_mat
        softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0).cuda()
        for _roiidx in range(retained.shape[0]):
            _weighted_feat = (
                softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
            )
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1).cuda()
            outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] += (
                _weighted_feat[0].detach().cuda().half()
            )
            outfeat[
                retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]
            ] = torch.nn.functional.normalize(
                outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]].float(),
                dim=-1,
            ).half()

        outfeat = outfeat.unsqueeze(0).float().cuda()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2).cuda()  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [512, 512], mode="nearest").cuda()
        outfeat = outfeat.permute(0, 2, 3, 1).cuda()  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1).cuda()
        outfeat = outfeat[0].half().cuda()  # --> H, W, feat_dim

        clip_end = time()
        clip_time += clip_end - clip_start
        torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)
    print(f"clip time: {clip_time}")


def blip_sam(save_dir_path, scene_dir_path, mask_dir_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = create_eva_vit_g(512, precision='fp32').to(device)
    for i in range(50):
        file = f"{i}_rgb.pt"
        INPUT_IMAGE_PATH = os.path.join(scene_dir_path, f"{i}_rgb.png")
        SEMIGLOBAL_FEAT_SAVE_FILE = os.path.join(save_dir_path, file)
        if os.path.isfile(SEMIGLOBAL_FEAT_SAVE_FILE):
            continue

        # read in the raw image
        raw_image = cv2.imread(INPUT_IMAGE_PATH)
        raw_image = cv2.resize(raw_image, (512, 512))
        image = torch.tensor(raw_image[:512, :512]).permute(2, 0, 1)
        image = image.unsqueeze(0).float().to(device)

        # run the image through the visual encoder to get a global feature vector
        output = visual_encoder(image)
        global_feat = torch.tensor(output)
        global_feat = global_feat.half().to(device)
        global_feat = global_feat.mean(1)
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)
        FEAT_DIM = global_feat.shape[-1]

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        # load the saved masks
        MASK_LOAD_FILE = os.path.join(mask_dir_path, file)
        outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, FEAT_DIM, dtype=torch.half)
        mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)
        mask = mask[:, :, :512, :512] # crop to 512x512
        num_masks = mask.shape[-3] # number of masks in the image

        rois = []
        roi_sim_per_unit_area = []
        feat_per_roi = []
        roi_nonzero_inds = []

        for _i in trange(num_masks):
            # get the mask for the current object and calculate the top left and bottom right corner of the bounding box around it
            curmask = mask[0, _i].long()
            bbox, nonzero_inds = get_bbox_around_mask(curmask)
            x0, y0, x1, y1 = bbox

            # calculate intersection over union to determine if the mask is too small; this happens if the SAM model picks up noise
            bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
            iou = bbox_area / img_area
            if iou < 0.005:
                continue

            # crop the image to the bounding box and run it through the visual encoder to get the feature vector for the object
            roi = torch.ones((512, 512, 3))
            img_roi = torch.tensor(raw_image[:512, :512])[x0:x1, y0:y1]
            roi[x0:x1, y0:y1] = img_roi
            img_roi = roi.permute(2, 0, 1).unsqueeze(0).to(device)
            roifeat = visual_encoder(img_roi)
            roifeat = torch.tensor(roifeat)
            roifeat = roifeat.half().cuda()
            roifeat = roifeat.mean(1)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)

            # calculate the cosine similarity between the global feature vector and the feature vector for the object and save that as well
            _sim = cosine_similarity(global_feat, roifeat)
            rois.append(torch.tensor(list(bbox)))
            roi_sim_per_unit_area.append(_sim)

        # run non-maximum suppression to get rid of overlapping masks
        rois = torch.stack(rois)
        scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
        retained = torchvision.ops.nms(rois.float().cpu(), scores.float().cpu(), iou_threshold=1.0)
        feat_per_roi = torch.cat(feat_per_roi, dim=0)

        print(f"retained {len(retained)} masks of {rois.shape[0]} total")
        retained_rois = rois[retained]
        retained_scores = scores[retained]
        retained_feat = feat_per_roi[retained]
        retained_nonzero_inds = []
        for _roiidx in range(retained.shape[0]):
            retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])

        # get the cosine similarity between the features of each object. This will be a square matrix where the (i, j)th entry is the cosine similarity between the ith and jth objects
        mask_sim_mat = torch.nn.functional.cosine_similarity(
            retained_feat[:, :, None], retained_feat.t()[None, :, :]
        )
        mask_sim_mat.fill_diagonal_(0.0) # set the diagonal to 0 because we don't want to consider the similarity between the same object
        mask_sim_mat = mask_sim_mat.mean(1)  # avg sim of each mask with each other mask
        softmax_scores = retained_scores.cuda() - mask_sim_mat # subtracting the object-object relevance (which can be thought of as the relevance of the object in context of the other objects) object-scene similarity (which is kind of like global relevance) gives how much more or less important that object is than all the other objects
        softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0) # apply softmax to get the final scores
        for _roiidx in range(retained.shape[0]):
            # weighted sum of the global feature vector and the feature vector for the object
            _weighted_feat = (
                softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
            )
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)

            # put the weighted feature vector back into the image at each pixel where the mask is nonzero,
            # creating the pixel-aligned object embeddings
            outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] += (
                _weighted_feat[0].detach().cpu().half()
            )
            outfeat[
                retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]
            ] = torch.nn.functional.normalize(
                outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]].float(),
                dim=-1,
            ).half()

        # make sure the pixel-aligned features are of size 512x512
        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [512, 512], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half()

        print(outfeat.shape)
        torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)

def fuse_features(save_dir, outputs, multiview_feat_dir):
    slam = PointFusion(odom="icp", dsratio=1, device="cuda", use_embeddings=True)

    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(
        device="cuda",
    )

    fusion_time = time()
    fovy = 45
    f = 0.5 * 512 / math.tan(fovy * math.pi / 360)
    intrinsics = np.array(((f, 0, 512 / 2,0), (0, f, 512 / 2,0), (0, 0, 1,0),(0,0,0,1)))
    intrinsics = torch.tensor(intrinsics)

    i = 0
    for file in ['sceneshotcam', 'apple', 'milk', 'cereal', 'bread', 'banana', 'bin', 'ur5e', 'panda', 'table']:
        rgb_image = outputs[file][0]
        rgb_image = torch.tensor(rgb_image)

        depth_image = outputs[file][1]
        depth_image = torch.tensor(depth_image)

        _color = rgb_image.float().cuda()  # Move color to CUDA
        _depth = depth_image.float().unsqueeze(-1).cuda()  # Move depth to CUDA
        # _pose = torch.from_numpy(outputs[file][3]).float().cuda()  # Move pose to CUDA

        _embedding = torch.load(os.path.join(multiview_feat_dir, f"{file}_rgb.pt"))
        _embedding = _embedding.float().cuda()  # Move embedding to CUDA right after load
        _embedding = torch.nn.functional.interpolate(
            _embedding.permute(2, 0, 1).unsqueeze(0),
            [512, 512],
            mode="nearest",
        )[0].permute(1, 2, 0).half().cuda()

        # load the RGBD image
        frame_cur = RGBDImages(
            _color.unsqueeze(0).unsqueeze(0).cuda(),
            _depth.unsqueeze(0).unsqueeze(0).cuda(),
            intrinsics.unsqueeze(0).unsqueeze(0).cuda(),
            # _pose.unsqueeze(0).unsqueeze(0),
            embeddings=_embedding.unsqueeze(0).unsqueeze(0).cuda(),
        )
        
        # run pointfusion, which will fuse the current frame with the previous frame
        # this also fuses the pixel aligned features with the pointcloud, which will allow it to be updated over time
        if i == 0:
            frame_prev = frame_cur.clone()
        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev)
        print(f"number of points: {pointclouds.points_padded.shape[1]}")
        frame_prev = frame_cur
        i += 1
        torch.cuda.empty_cache()

    print(f"fuse time: {time() - fusion_time}")

    pointclouds.save_to_h5(save_dir)

from time import time
def main():
    import pickle
    with open("outputs.pkl", "rb") as f:
        outputs = pickle.load(f)

    mask_dir = os.path.join(os.getcwd(), "sam_masks")
    # os.makedirs(mask_dir, exist_ok=True)
    # create_masks(mask_dir, outputs)

    multiview_feat_dir = os.path.join(os.getcwd(), "multiview_features")
    # os.makedirs(multiview_feat_dir, exist_ok=True)
    # clip_start = time()
    # clip_sam(multiview_feat_dir, mask_dir, outputs)
    # clip_end = time()
    # print(f"clip time: {clip_end - clip_start}")

    fused_3d_feat_dir = os.path.join(os.getcwd(), "fused_3d_features")
    os.makedirs(fused_3d_feat_dir, exist_ok=True)
    fuse_start = time()
    fuse_features(fused_3d_feat_dir, outputs, multiview_feat_dir)
    fuse_end = time()
    print(f"fuse time: {fuse_end - fuse_start}")

if __name__ == "__main__":
    main()


    


    

    

    