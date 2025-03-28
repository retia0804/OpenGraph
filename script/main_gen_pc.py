"""
2024.01.18
读取离线的caption和mask，映射到点云
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "thirdparties/4DMOS/src",
    )
)

import numpy as np
import cv2
import torch
from pathlib import Path
import gzip
import pickle
from sentence_transformers import SentenceTransformer
from datetime import datetime
from utils.utils import (
    project,
    gobs_to_detection_list,
    denoise_objects,
    filter_objects,
    merge_objects,
    accumulate_pc,
    distance_filter,
    show_captions,
    class_objects,
)
from tqdm import trange
from some_class.datasets_class import SemanticKittiDataset
from some_class.map_calss import MapObjectList
from utils.merge import (
    compute_spatial_similarities,
    compute_caption_similarities,
    compute_ft_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    caption_merge,
    captions_ft,
)
import open3d as o3d
import hydra
from omegaconf import DictConfig
from utils.merge import merge_obj2_into_obj1
from mos4d.mos4d import MOS4DNet
from mos4d.config import MOS4DConfig

# 一些背景常见的caption：道路，人行道
# 背景可能产生的标签
BG_CAPTIONS = [
    "paved city road",
    "a long narrow street",
    "paved city street",
    "white lines on the road",
    "shadows on the street",
    "a concerte sidewalk",
    "a shadow on the ground",
    "white lines on the street",
    "brick sidewalk",
    "a sidewalk next to the street",
    "a train boarding platform",
    "the train tracks",
    "shadow of fence",
    "a sidewalk next to the train tracks",
    "shadow of bench",
    "a paved city sidewalk",
]
# 背景映射出来的标签
BG_CAPTIONS_Pro = [
    "paved road",
    "paved road",
    "paved road",
    "paved road",
    "paved road",
    "paved road",
    "paved road",
    "paved road",
    "sidewalk",
    "sidewalk",
    "sidewalk",
    "sidewalk",
    "sidewalk",
    "sidewalk",
    "sidewalk",
    "sidewalk",
]
# 背景映射出来的标签有哪些
BG_CAPTIONS_Pro_Sim = ["paved road", "sidewalk"]


def process_cfg(cfg: DictConfig):
    """
    配置文件预处理
    """
    cfg.basedir = Path(cfg.basedir)
    cfg.save_vis_path = Path(cfg.save_vis_path)
    cfg.save_cap_path = Path(cfg.save_cap_path)
    cfg.save_pcd_path = Path(cfg.save_pcd_path)
    return cfg


def convert_to_mos4d_config(original_cfg):
    """
    원본 설정을 MOS4DConfig 형식으로 변환
    """
    # 기본 설정값
    mos4d_cfg = {
        "data": {"deskew": False, "max_range": 100.0, "min_range": 3.0},
        "odometry": {
            "voxel_size": original_cfg.get("DATA", {}).get("VOXEL_SIZE", 0.5),
            "max_points_per_voxel": 20,
            "initial_threshold": 2.0,
            "min_motion_th": 0.1,
        },
        "mos": {
            "voxel_size_mos": original_cfg.get("DATA", {}).get("VOXEL_SIZE", 0.1),
            "delay_mos": original_cfg.get("MODEL", {}).get("N_PAST_STEPS", 10),
            "prior": 0.25,
            "max_range_mos": 50.0,
            "min_range_mos": 0.0,
        },
        "training": {
            "id": original_cfg.get("EXPERIMENT", {}).get("ID", "experiment_id"),
            "train": [
                str(x)
                for x in original_cfg.get("DATA", {}).get("SPLIT", {}).get("TRAIN", [])
            ],
            "val": [
                str(x)
                for x in original_cfg.get("DATA", {}).get("SPLIT", {}).get("VAL", [])
            ],
            "batch_size": original_cfg.get("TRAIN", {}).get("BATCH_SIZE", 16),
            "accumulate_grad_batches": original_cfg.get("TRAIN", {}).get(
                "ACC_BATCHES", 1
            ),
            "max_epochs": original_cfg.get("TRAIN", {}).get("MAX_EPOCH", 100),
            "lr": original_cfg.get("TRAIN", {}).get("LR", 0.0001),
            "lr_epoch": original_cfg.get("TRAIN", {}).get("LR_EPOCH", 1),
            "lr_decay": original_cfg.get("TRAIN", {}).get("LR_DECAY", 0.99),
            "weight_decay": original_cfg.get("TRAIN", {}).get("WEIGHT_DECAY", 0.0001),
            "num_workers": original_cfg.get("DATA", {}).get("NUM_WORKER", 4),
        },
    }
    return mos4d_cfg


@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg: DictConfig):
    # 先处理一下cfg
    cfg = process_cfg(cfg)
    # 加载所使用的数据集
    datasets = SemanticKittiDataset(
        cfg.basedir, cfg.sequence, stride=cfg.stride, start=cfg.start, end=cfg.end
    )
    print("Load a dataset with a size of:", len(datasets))
    # 初始化地图
    objects = MapObjectList(device="cuda")
    # 是否过滤动态目标，如果需要就加载模型
    mos_model = None
    if cfg.filter_dynamic:
        weights = cfg.mos_path
        mos_cfg = torch.load(weights)["hyper_parameters"]
        # 转换配置格式
        mos_cfg = convert_to_mos4d_config(mos_cfg)
        config = MOS4DConfig(**mos_cfg)
        model = MOS4DNet(config.mos.voxel_size_mos)
        state_dict = torch.load(weights)["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("mos.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
        model.load_state_dict(state_dict)
        mos_model = model.cuda()
        mos_model.eval()
        mos_model.freeze()
    # 单独处理背景为一个整体
    if cfg.use_bg:
        bg_objects = {c: None for c in BG_CAPTIONS_Pro_Sim}
        # 加载SBERT模型
        sbert_model = SentenceTransformer(cfg.sbert_path)
        sbert_model = sbert_model.to("cuda")
        # 把这些背景的caption先编码了
        bg_fts = []
        for bg_cation in BG_CAPTIONS:
            bg_ft = sbert_model.encode(bg_cation, convert_to_tensor=True)
            bg_ft = bg_ft / bg_ft.norm(dim=-1, keepdim=True)
            bg_ft = bg_ft.squeeze()
            bg_fts.append(bg_ft)
    else:
        bg_objects = None
    point_clouds = []
    for idx in trange(len(datasets)):
        # 第0帧点云太稀疏了，不要了，并且无法估计动态目标
        if idx == 0:
            continue
        image, pc, pose, his_pcs, his_poses = datasets[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 对历史帧的点云进行累积，顺便滤除动态物体
        if his_pcs is not None:
            pc = accumulate_pc(cfg, mos_model, pc, pose, his_pcs, his_poses)
        # 过滤掉距离值太远的点云，可不用
        if cfg.filter_dis:
            pc = distance_filter(cfg.max_depth, pc)
        # 得到点云投影后的像素值和剩下的点云，是一一对应的
        pro_point, pixels = project(pc, image, datasets.calib)
        file_names = os.path.basename(datasets.color_paths[idx])
        save_path_cap = Path(
            os.path.join(cfg.save_cap_path, f"cap_{file_names}")
        ).with_suffix(".pkl.gz")
        # 打开保存的文件
        with gzip.open(save_path_cap, "rb") as f:
            gobs = pickle.load(f)
        # 得到当前帧的DetectionList类型的点云
        detection_list, bg_list = gobs_to_detection_list(
            cfg=cfg,
            image=image,
            pc=pro_point,
            pixels=pixels,
            idx=idx,
            gobs=gobs,
            trans_pose=pose,
            bg_fts=bg_fts,
            BG_CAPTIONS_Pro=BG_CAPTIONS_Pro,
        )
        # 先单独处理背景
        if len(bg_list) > 0:
            for detected_object in bg_list:
                class_name = detected_object["bg_class"]
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    bg_objects[class_name] = merge_obj2_into_obj1(
                        cfg, matched_obj, matched_det, bg=True, class_name=class_name
                    )
        # 如果没有值得加入全局地图的点云
        if len(detection_list) == 0:
            continue
        # 如果是第一帧，全部加入
        if len(objects) == 0:
            for i in range(len(detection_list)):
                objects.append(detection_list[i])
            # 并且跳过下面的相似度计算
            continue
        # 可视化一下
        if cfg.vis_all:
            point_clouds.extend(
                [detection_list[i]["pcd"] for i in range(len(detection_list))]
            )
        # 计算相似度
        spatial_sim = compute_spatial_similarities(detection_list, objects)
        caption_sim = compute_caption_similarities(detection_list, objects)
        ft_sim = compute_ft_similarities(detection_list, objects)
        agg_sim = aggregate_similarities(cfg, spatial_sim, ft_sim, caption_sim)
        # DEBUG: 相似性判断
        # debug_sim = np.dstack((spatial_sim, caption_sim,ft_sim,agg_sim))
        # for i in range(debug_sim.shape[0]):
        #     for j in range(debug_sim.shape[1]):
        #         # 只看有重叠的
        #         if (debug_sim[i][j][0]>0):
        #             print(detection_list[i]["caption"], "***VS***",objects[j]["caption"],debug_sim[i][j])
        # 设置阈值判断是否一个物体。如果低于阈值，则设置为负无穷大
        agg_sim[agg_sim < cfg.sim_threshold] = float("-inf")
        # 按照相似性融合
        objects = merge_detections_to_objects(cfg, detection_list, objects, agg_sim)
    if cfg.vis_all:
        o3d.visualization.draw_geometries(point_clouds)

    # 构建完地图之后，降采样地图降分辨率，去噪一下
    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        bg_objects = denoise_objects(cfg, bg_objects, bg=True)
    objects = denoise_objects(cfg, objects)
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    # show_captions(objects, bg_objects)
    # 根据最后的结果，融合物体
    objects, generator = caption_merge(cfg, objects)
    # 最后再计算一下融合的caption的特征
    if cfg.caption_merge_ft:
        objects, bg_objects = captions_ft(objects, bg_objects, sbert_model)
    # show_captions(objects, bg_objects)
    # 根据最后的结果，分类得到class
    objects, bg_objects = class_objects(
        cfg, sbert_model, objects, bg_objects, generator
    )
    show_captions(objects, bg_objects)

    # 保存一个处理的地图
    if cfg.save_pcd:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "objects": objects.to_serializable(),
            "bg_objects": None if bg_objects is None else bg_objects.to_serializable(),
            "cfg": cfg,
        }
        pcd_save_path = cfg.save_pcd_path / f"full_pcd.pkl.gz"
        # 如果目录不存在则创建
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"保存点云地图到 {pcd_save_path}")


if __name__ == "__main__":
    main()
