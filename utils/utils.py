"""
2024.01.17
工具函数文件
"""

import sys
import copy

sys.path.append("/code1/dyn/github_repos/OpenGraph")
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
from tokenize_anything import model_registry

sys.path.append("/home/dyn/multimodal/Grounded-Segment-Anything")
sys.path.append("/home/dyn/multimodal/Grounded-Segment-Anything/Tag2Text")
sys.path.append("/code1/dyn/github_repos/OpenGraph")
from some_class.amg_class import MyAutomaticMaskGenerator
from some_class.map_calss import DetectionList
import open3d as o3d
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import spacy
from some_class.map_calss import MapObjectList
import torch.nn.functional as F
import json
from llama import Llama, Dialog
import faiss
import re
import openai
from tqdm import trange

try:
    from ram.models import tag2text
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your PATH. ")
    raise e

try:
    from groundingdino.util.inference import Model
except ImportError as e:
    print(
        "Import Error: Please install Grounded Segment Anything following the instructions in README."
    )
    raise e


# spacy分词时的一些先验词汇表
CONFUSED_NOUNS = ["metal", "back", "part", "row", "triangular", "patch"]
INTEREST_NOUNS = ["van", "house"]
INTEREST_ADJS = ["grassy", "white"]


def load_models(cfg):
    """
    加载各个模型，传入三个大模型+SBERT
    """
    # 加载一下ram模型
    TAG2TEXT_CHECKPOINT_PATH = cfg.tag2text_path
    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)
    # load model
    tagging_model = tag2text(
        pretrained=TAG2TEXT_CHECKPOINT_PATH,
        image_size=384,
        vit="swin_b",
        delete_tag_index=delete_tag_index,
    )
    tagging_model = tagging_model.eval().to("cuda")
    # dino模型分割
    grounding_dino_model = Model(
        model_config_path=cfg.gd_path,
        model_checkpoint_path=cfg.gd_weights,
        device="cuda",
    )
    # 使用模型和图像创建分割器
    model_type = "tap_vit_l"
    checkpoint = cfg.tap_path
    tap_model = model_registry[model_type](checkpoint=checkpoint)
    concept_weights = cfg.tap_merge_path
    tap_model.concept_projector.reset_weights(concept_weights)
    tap_model.text_decoder.reset_cache(max_batch_size=1000)

    # SBERT文本编码器
    sbert_model = SentenceTransformer(cfg.sbert_path)

    # 创建分割器
    mask_generator = MyAutomaticMaskGenerator(
        tagging_model=tagging_model,
        grounding_dino_model=grounding_dino_model,
        tap_model=tap_model,
        sbert_model=sbert_model,
    )
    print("\n 恭喜！所有大模型加载完成，可任意使用！\n")
    return mask_generator


def project(points, image, calib):
    """
    把点云投影到图像上，输入点云，输出点云对应的图像横纵坐标
    """
    points_homo = np.insert(points, 3, 1, axis=1).T
    pointCloud = np.delete(points, np.where(points_homo[0, :] < 0), axis=0)
    # 以列为基准, 删除深度x=0的点
    points_homo = np.delete(points_homo, np.where(points_homo[0, :] < 0), axis=1)
    # 相机坐标系3D点=相机02内参*雷达到激光的变换矩阵*雷达3D点
    proj_lidar = calib["P_rect_20"].dot(calib["T_cam2_velo"]).dot(points_homo)
    # 以列为基准, 删除投影图像点中深度z<0(在投影图像后方)的点 #3xN
    cam = np.delete(proj_lidar, np.where(proj_lidar[2, :] < 0), axis=1)
    pointCloud = np.delete(pointCloud, np.where(proj_lidar[2, :] < 0), axis=0)
    # 前两行元素分布除以第三行元素(归一化到相机坐标系z=1平面)(x=x/z, y =y/z)
    cam[:2, :] /= cam[2, :]
    # 投影到图像
    IMG_H, IMG_W, _ = image.shape
    # 过滤掉不在相机上的
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    points = np.delete(pointCloud, np.where(outlier), axis=0)
    u, v, z = cam
    pixels = np.dstack((v, u)).squeeze()
    return points, pixels


def create_object_pcd(
    image, pc, pixels, mask, obj_color=None
) -> o3d.geometry.PointCloud:
    """
    得到rgb的点云
    """
    mask_for_pc = mask[pixels[:, 0].astype(int), pixels[:, 1].astype(int)]
    points = pc[mask_for_pc]
    pixels = pixels[mask_for_pc]
    colors = image[pixels[:, 0].astype(int), pixels[:, 1].astype(int)] / 255.0
    # 对点稍加扰动以避免共线性，看不出来是雷达线条
    points += np.random.normal(0, 4e-3, points.shape)
    # 创建一个Open3D PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pcd_denoise_dbscan(
    pcd: o3d.geometry.PointCloud, eps=0.01, min_points=10
) -> o3d.geometry.PointCloud:
    """
    通过聚类，移除噪点
    """
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    # 转换为 numpy 数组
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)
    # 统计群组中的所有标签
    counter = Counter(pcd_clusters)
    # 去除噪音标签
    if counter and (-1 in counter):
        del counter[-1]
    if counter:
        # 找出最大集群的标签
        most_common_label, _ = counter.most_common(1)[0]
        # 为最大群组中的点创建掩码
        largest_mask = pcd_clusters == most_common_label
        # 应用 mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        # 如果最大群集太小，则返回原始点云
        if len(largest_cluster_points) < 5:
            return pcd
        # 创建新的 PointCloud 对象
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        pcd = largest_cluster_pcd
    return pcd


def process_pcd(cfg, pcd, use_db=True):
    """
    对pcd降噪处理，剔除离群点
    """
    # 以2.5cm降采样
    pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)
    # dubug，降噪前后对比
    # o3d.visualization.draw_geometries([pcd])
    # 进行降噪，但是这种降噪方式对于点云不太好
    if cfg.dbscan_remove_noise and use_db:
        # cl, index = pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
        # pcd = pcd.select_by_index(index)
        pcd = pcd_denoise_dbscan(
            pcd, eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points
        )
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def get_bounding_box(pcd):
    """
    得到点云的bbox
    """
    # 推荐使用定向的
    if len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()


def gobs_to_detection_list(
    cfg,
    image,
    pc,
    pixels,
    idx,
    gobs,
    trans_pose=None,
    bg_fts=None,
    BG_CAPTIONS_Pro=None,
):
    """
    从gobs返回一个DetectionList对象,所有对象在当前帧中。
    """
    detection_lists = DetectionList()
    bg_list = DetectionList()
    # 没有数据则返回空的
    if len(gobs) == 0:
        return detection_lists, bg_list
    n_masks = len(gobs)
    # 对每个mask处理
    for mask_idx in range(n_masks):
        mask = gobs[mask_idx]["mask"].squeeze()
        caption = gobs[mask_idx]["caption"]
        caption_ft = gobs[mask_idx]["caption_ft"]
        # 得到pcd
        camera_object_pcd = create_object_pcd(image, pc, pixels, mask, obj_color=None)
        # 实例设置随机颜色
        color = np.random.random(3)
        # 这个对象最少得5个点吧，否则不要也罢
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):
            continue
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        # 获取最大群，以过滤噪音
        global_object_pcd = process_pcd(cfg, global_object_pcd)
        pcd_bbox = get_bounding_box(global_object_pcd)
        pcd_bbox.color = [0, 1, 0]
        # 如果物体太小了，也要删掉
        if pcd_bbox.volume() < 1e-6:
            continue
        bg_class = None
        # 如果使用背景的话，比较与背景的相似度是否超过阈值
        if cfg.use_bg:
            caption_ft_cuda = caption_ft.to("cuda")
            for i in range(len(bg_fts)):
                similarity = F.cosine_similarity(bg_fts[i], caption_ft_cuda, dim=-1)
                if similarity > cfg.bg_rate:
                    bg_class = BG_CAPTIONS_Pro[i]
                    # 有高的就直接跳出去
                    break
        # 把这个物体存储下来吧
        detected_object = {
            "image_idx": [idx],  # 哪个图像看到的，注意stride倍数关系
            "num_detections": 1,  # 这个物体的检测数量，现在一个物体所以是1
            "n_points": len(global_object_pcd.points),  # 这个物体的点数量
            "inst_color": color,  # 该段实例使用的随机颜色，后面按照semantickitti赋值
            "bg_class": bg_class,  # 该段实例属于哪个背景，不是背景则不需要
            # 下面这些事针对全局地图中物体的，因为这个有可能是新物体
            "class_sk": None,  # 该实例的类别，后面可视化会用
            "caption": caption,  # 该实例的caption，后面会融合
            "captions_ft": None,  # 该实例的融合后的caption的编码特征
            "ft": caption_ft,  # caption的编码结果
            "pcd": global_object_pcd,  # 点云pcd
            "bbox": pcd_bbox,  # 该实例的bbox
        }
        # 分类归纳
        if cfg.use_bg and bg_class is not None:
            bg_list.append(detected_object)
        else:
            detection_lists.append(detected_object)
    return detection_lists, bg_list


def denoise_objects(cfg, objects: MapObjectList, bg=False):
    """
    整个地图去噪
    """
    for i in range(len(objects)):
        og_object_pcd = objects[i]["pcd"]
        if bg:
            objects[i]["pcd"] = process_pcd(cfg, objects[i]["pcd"], use_db=False)
        else:
            objects[i]["pcd"] = process_pcd(cfg, objects[i]["pcd"], use_db=True)
        if len(objects[i]["pcd"].points) < 4:
            objects[i]["pcd"] = og_object_pcd
            continue
        objects[i]["bbox"] = get_bounding_box(objects[i]["pcd"])
        objects[i]["bbox"].color = [0, 1, 0]
    return objects


def filter_objects(cfg, objects: MapObjectList):
    """
    后处理，移除掉太少点云的物体和太少观测的物体
    """
    print("Before final map filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if (
            len(obj["pcd"].points) >= cfg.obj_min_points
            and obj["num_detections"] >= cfg.obj_min_detections
        ):
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After final map filtering: ", len(objects))
    return objects


def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    """
    计算3diou的占比，太小了就不进行评判最后的融合了
    """
    # 获取第一个包围盒的坐标
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding
    # 获取第二个包围盒的坐标
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding
    # 计算两个边界框的重叠部分
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)
    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)
    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)
    if use_iou:
        return iou
    else:
        return max_overlap


def compute_overlap_matrix(cfg, objects: MapObjectList):
    """
    用最近邻点计算对象间的成对重叠。假设我们有一个包含n个点云的列表，每个点云都是一个o3d.geometry.PointCloud对象。
    现在，我们要构建一个大小为nxn的矩阵，其中(i, j)条目是点云i中的点与任意一个点的距离在阈值范围内的点与点云j中任意点的距离阈值的比率。
    """
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    # 将点云转换为 numpy 数组，然后转换为 FAISS 索引，以便高效搜索
    point_arrays = [np.asarray(obj["pcd"].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    # 将 numpy 数组中的点添加到相应的 FAISS 索引中
    for index, arr in zip(indices, point_arrays):
        index.add(arr)
    # 计算成对重叠
    for i in range(n):
        for j in range(n):
            if i != j:  # 跳过对角线元素
                box_i = objects[i]["bbox"]
                box_j = objects[j]["bbox"]
                # 如果方框完全不重叠，则跳过（节省计算）
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                # 使用range_search查找阈值范围内的点
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)
                # 如果在阈值范围内发现任何点，则增加重叠计数
                # overlap += sum([len(i) for i in I])
                overlap = (D < cfg.voxel_size**2).sum()  # D 是距离的平方
                # 计算阈值内点的比率
                overlap_matrix[i, j] = overlap / len(point_arrays[i])
    return overlap_matrix


def to_numpy(tensor):
    """
    转为numpy
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(numpy_array, device=None):
    """
    转为tensor
    """
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)


def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    """
    最后后处理，融合重叠物体
    """
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]
    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        ft_sim = F.cosine_similarity(
            to_tensor(objects[i]["ft"]), to_tensor(objects[j]["ft"]), dim=0
        )
        if ratio > cfg.merge_overlap_thresh and ft_sim > cfg.merge_ft_thresh:
            if kept_objects[j]:
                # 然后将对象 i 并入对象 j
                from utils.merge import merge_obj2_into_obj1

                objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i])
                kept_objects[i] = False
        else:
            break
    # 删除已合并的对象
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    return objects


def merge_objects(cfg, objects: MapObjectList):
    """
    后处理，最后融合一次重叠度太高的
    """
    if cfg.merge_final:
        overlap_matrix = compute_overlap_matrix(cfg, objects)
        print("Before final map fusion:", len(objects))
        objects = merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After Final Map Fusion:", len(objects))
    return objects


def transform_point_cloud(past_point_clouds, from_pose, to_pose):
    """
    점운을 좌표 변환합니다. numpy 배열 또는 torch 텐서를 입력으로 받습니다.
    Args:
        past_point_clouds: 변환할 점군 (N, 3) 또는 (N, 4) 또는 (N, 5)
        from_pose: 원본 좌표계의 변환 행렬 (4, 4)
        to_pose: 목표 좌표계의 변환 행렬 (4, 4)
    Returns:
        변환된 점군 (N, 3)
    """
    # numpy 배열인 경우 torch 텐서로 변환
    if isinstance(past_point_clouds, np.ndarray):
        past_point_clouds = torch.from_numpy(past_point_clouds.astype(np.float32))

    transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
    NP = past_point_clouds.shape[0]

    # 점군이 3차원보다 큰 경우 (x,y,z,...)
    if past_point_clouds.shape[1] > 3:
        past_point_clouds = past_point_clouds[:, :3]

    # 점군을 동차 좌표로 변환 (Nx3 -> Nx4)
    xyz1 = torch.hstack(
        [past_point_clouds, torch.ones(NP, 1).type_as(past_point_clouds)]
    )

    # 변환 행렬과 점군을 곱함 (4x4 @ Nx4 -> Nx4)
    transformed = (transformation @ xyz1.T).T

    # 동차 좌표에서 3D 좌표로 변환 (Nx4 -> Nx3)
    transformed = transformed[:, :3]

    return transformed


def timestamp_tensor(tensor, time):
    """
    增加时间作为增加的一列，用与判断动态与否
    """
    n_points = tensor.shape[0]
    time = time * torch.ones((n_points, 1))
    timestamped_tensor = torch.hstack([tensor, time])
    return timestamped_tensor


def accumulate_pc(cfg, mos_model, pc, pose, his_pcs, his_poses):
    """
    누적된 포인트 클라우드를 생성하고 동적 객체를 필터링합니다.
    Args:
        cfg: 설정
        mos_model: 동적 객체 감지 모델
        pc: 현재 프레임의 포인트 클라우드 (N, 3)
        pose: 현재 프레임의 포즈
        his_pcs: 이전 프레임들의 포인트 클라우드 리스트
        his_poses: 이전 프레임들의 포즈 리스트
    Returns:
        누적된 포인트 클라우드
    """
    # 현재 프레임의 포인트 클라우드를 변환 (월드 좌표계로)
    pc = transform_point_cloud(pc, pose, np.eye(4))

    # 이전 프레임들의 포인트 클라우드를 현재 프레임으로 변환
    past_point_clouds = []
    for i, (his_pc, his_pose) in enumerate(zip(his_pcs, his_poses)):
        # 포인트 클라우드 변환 (현재 프레임 좌표계로)
        transformed_pc = transform_point_cloud(his_pc, his_pose, pose)
        # 시간 차원 추가 (N, 3) -> (N, 4)
        time_value = round((i - cfg.stride + 1) * 0.1, 3)
        time_dim = torch.ones(len(transformed_pc), 1) * time_value
        transformed_pc = torch.cat([transformed_pc, time_dim], dim=1)
        past_point_clouds.append(transformed_pc)

    # 모든 포인트 클라우드를 하나로 합침
    if len(past_point_clouds) > 0:
        past_point_clouds = torch.cat(past_point_clouds, dim=0)
        print(f"Past point clouds shape: {past_point_clouds.shape}")
        # 동적 객체 필터링
        if cfg.filter_dynamic and mos_model is not None:
            # 포인트 클라우드를 CUDA로 이동
            past_point_clouds = past_point_clouds.float().cuda()
            # 동적 객체 예측
            logits = mos_model.predict(past_point_clouds)
            # 동적 객체 마스크 생성
            mask = logits < 0
            # 정적 객체만 선택
            past_point_clouds = past_point_clouds[mask, :3].cpu().numpy()
    else:
        past_point_clouds = np.zeros((0, 3))

    # 현재 프레임과 이전 프레임의 포인트 클라우드 합치기
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().numpy()
    accumulated_pc = np.concatenate([pc, past_point_clouds], axis=0)
    return accumulated_pc


def distance_filter(max_depth, pc):
    """
    过滤掉深度值太大的点云
    """
    # 计算每个点的距离
    distances = np.linalg.norm(pc, axis=1)
    # 筛选出距离在max_depth之内的点
    filtered_points = pc[distances <= max_depth]
    return filtered_points


def caption_extract(idx, spacy_nlp, caption_ori):
    # 使用spaCy处理句子
    doc = spacy_nlp(str(caption_ori))
    tokens = [token.text for token in doc]
    main_noun = "none"
    main_adj = []
    extra_captions = []

    # 获取名词
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    # 获取形容词
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]

    # 将名词属性中的第一个名词视为关键词 但是要排除一些混淆概念in confused_nouns
    for i in range(len(nouns)):
        if nouns[i] not in CONFUSED_NOUNS:
            main_noun = nouns[i]
            break
    for token in tokens:
        if token in INTEREST_NOUNS:
            main_noun = token
            break

    # 记录当前提取的主语在token中idx
    main_noun_idx = tokens.index(main_noun)

    # 只有idx位于主语之前的形容词保留
    for adj in adjectives:
        if (tokens.index(adj) < main_noun_idx) and (len(main_adj) < 2):
            main_adj += [adj]

    # 如果没有提取到有效的形容词 那得看看是不是遗漏了一些
    if not main_adj:
        for token in tokens:
            if (
                token in INTEREST_ADJS
                and (len(main_adj) < 2)
                and (tokens.index(token) < main_noun_idx)
            ):
                main_adj += [token]

    extra_captions = main_adj + [main_noun]
    extra_captions = " ".join(extra_captions)
    # print(f"Extracted Captions {idx} as: {extra_captions}")
    return extra_captions


def class_objects(
    cfg, sbert_model, objects: MapObjectList, bg_objects: MapObjectList, generator
):
    """
    按照caption和ft给物体分类semantickitti的类别，并设置inst_color
    """
    # 加载语义和颜色文件
    file_path = cfg.class_colors_json
    with open(file_path, "r") as json_file:
        class_colors_sk_disk = json.load(json_file)
        class_names_sk = list(class_colors_sk_disk.keys())
        class_colors_sk = list(class_colors_sk_disk.values())
        class_colors_sk = [
            list(map(lambda x: x / 255.0 if isinstance(x, (int, float)) else x, color))
            for color in class_colors_sk
        ]
    if (
        cfg.class_methods == "sbert"
        or cfg.class_methods == "llama"
        or cfg.class_methods == "gpt"
    ):
        # 计算所有class的特征，不仅sbert用，对于llama以及gpt没有输出对的也能用
        class_name_fts = None
        for class_name in class_names_sk:
            class_name_ft = sbert_model.encode(class_name, convert_to_tensor=True)
            class_name_ft = class_name_ft / class_name_ft.norm(dim=-1, keepdim=True)
            class_name_ft = class_name_ft.squeeze()
            if class_name_fts is None:
                class_name_fts = class_name_ft
            else:
                class_name_fts = torch.vstack((class_name_fts, class_name_ft))
    if cfg.class_methods == "sbert":
        # 是否先spacy分词在进行相似性判断
        if cfg.spacy:
            # 加载英语模型
            spacy_nlp = spacy.load("en_core_web_sm")
            print("Spacy English loaded successfully! Ready for caption extraction!")
        # 为每个物体找最相似的语义类别，记录颜色
        for i in trange(len(objects)):
            # 先使用最后的caption计算特征
            caption = objects[i]["caption"]
            if cfg.spacy:
                caption = caption_extract(i, spacy_nlp, caption)
            caption_only_ft = sbert_model.encode(caption, convert_to_tensor=True)
            caption_only_ft = caption_only_ft / caption_only_ft.norm(
                dim=-1, keepdim=True
            )
            caption_only_ft = caption_only_ft.squeeze()
            # 再使用融合的caption_ft
            objects_sbert_fts = objects[i]["ft"]
            objects_sbert_fts = objects_sbert_fts.to("cuda")
            # 两个加权融合
            final_ft = (
                caption_only_ft * cfg.vis_caption_weight
                + objects_sbert_fts * cfg.vis_ft_weight
            )
            # 与class计算相似性
            similarities = F.cosine_similarity(
                class_name_fts, final_ft.unsqueeze(0), dim=-1
            )
            if cfg.spacy and cfg.caption_only:
                similarities = F.cosine_similarity(
                    class_name_fts, caption_only_ft.unsqueeze(0), dim=-1
                )
            max_indices = torch.argmax(similarities)
            # 设置好类别和颜色
            objects[i]["class_sk"] = class_names_sk[max_indices]
            objects[i]["inst_color"] = class_colors_sk[max_indices]
        if bg_objects is not None:
            for i in trange(len(bg_objects)):
                # 先使用最后的caption计算特征
                caption = bg_objects[i]["caption"]
                if cfg.spacy:
                    caption = caption_extract(i, spacy_nlp, caption)
                caption_only_ft = sbert_model.encode(caption, convert_to_tensor=True)
                caption_only_ft = caption_only_ft / caption_only_ft.norm(
                    dim=-1, keepdim=True
                )
                caption_only_ft = caption_only_ft.squeeze()
                # 再使用融合的caption_ft
                objects_sbert_fts = bg_objects[i]["ft"]
                objects_sbert_fts = objects_sbert_fts.to("cuda")
                # 两个加权融合
                final_ft = caption_only_ft * 0.5 + objects_sbert_fts * 0.5
                # 与class计算相似性
                similarities = F.cosine_similarity(
                    class_name_fts, final_ft.unsqueeze(0), dim=-1
                )
                if cfg.spacy and cfg.caption_only:
                    similarities = F.cosine_similarity(
                        class_name_fts, caption_only_ft.unsqueeze(0), dim=-1
                    )
                max_indices = torch.argmax(similarities)
                # 设置好类别和颜色
                bg_objects[i]["class_sk"] = class_names_sk[max_indices]
                bg_objects[i]["inst_color"] = class_colors_sk[max_indices]
    elif cfg.class_methods == "llama":
        # 用作示范的prompt example
        caption_example1 = "a car parked on the street"
        caption_example2 = "a red and white sign"
        caption_example3 = "grass on the side of the road"
        caption_example4 = "a sign on a pole"
        DEFAULT_PROMPT = """
        You are a classifier that can categorize a caption phrase into one of the following categories based on a caption phrase.
        List of categories: [car, bicycle, motorcycle, truck, person, bicyclist, motorcyclist, road,
        parking, sidewalk, building, fence, vegetation, trunk, terrain, pole, traffic-sign].
        You only need to generate one category name which must be included in this list. 
        The output format is 'Category name: [[your summarized category name itself]]'
        Emphasizing again: Do not provide words beyond the given list!!! Please test it yourself and regenerate it if it exceeds the list.
        """
        for i in trange(len(objects)):
            caption_obj = objects[i]["caption"]
            # 生成llama对话
            dialogs: List[Dialog] = [
                [
                    {"role": "system", "content": DEFAULT_PROMPT},
                    {"role": "user", "content": caption_example1},
                    {"role": "assistant", "content": "Category name: [car]"},
                    {"role": "user", "content": caption_example2},
                    {"role": "assistant", "content": "Category name: [traffic-sign]"},
                    {"role": "user", "content": caption_example3},
                    {"role": "assistant", "content": "Category name: [terrain]"},
                    {"role": "user", "content": caption_example4},
                    {"role": "assistant", "content": "Category name: [traffic-sign]"},
                    {"role": "user", "content": caption_obj},
                ],
            ]
            # llama进行回答
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=None,
                temperature=0.6,
                top_p=0.9,
            )
            # 读取llama回答结果中的generation content作为caption融合结果
            for dialog, result in zip(dialogs, results):
                input_text = result["generation"]["content"]
                pattern = r"\[([^]]+)\]"  # 匹配方括号中的内容
                match = re.search(pattern, input_text)
                extracted_content = ""  # 빈 문자열로 초기화
                if match:
                    extracted_content = match.group(1)
                else:
                    # match가 없을 경우 sbert를 사용하여 처리
                    caption_ft = sbert_model.encode(input_text, convert_to_tensor=True)
                    caption_ft = caption_ft / caption_ft.norm(dim=-1, keepdim=True)
                    caption_ft = caption_ft.squeeze()
                    similarities = F.cosine_similarity(
                        class_name_fts, caption_ft.unsqueeze(0), dim=-1
                    )
                    max_indices = torch.argmax(similarities)
                    extracted_content = class_names_sk[max_indices]
            # 如果llama生成的特征没有在给定列表中，则使用sbert特征配准
            if extracted_content not in class_colors_sk_disk:
                extracted_content_ft = sbert_model.encode(
                    extracted_content, convert_to_tensor=True
                )
                extracted_content_ft = extracted_content_ft / extracted_content_ft.norm(
                    dim=-1, keepdim=True
                )
                extracted_content_ft = extracted_content_ft.squeeze()
                # 与class计算相似性
                similarities = F.cosine_similarity(
                    class_name_fts, extracted_content_ft.unsqueeze(0), dim=-1
                )
                max_indices = torch.argmax(similarities)
                # 设置好类别和颜色
                objects[i]["class_sk"] = class_names_sk[max_indices]
                objects[i]["inst_color"] = class_colors_sk[max_indices]
            else:
                objects[i]["class_sk"] = extracted_content
                objects[i]["inst_color"] = (
                    np.array(class_colors_sk_disk[extracted_content]) / 255.0
                )
        if bg_objects is not None:
            for i in trange(len(bg_objects)):
                caption_obj = bg_objects[i]["caption"]
                # 生成llama对话
                dialogs: List[Dialog] = [
                    [
                        {"role": "system", "content": DEFAULT_PROMPT},
                        {"role": "user", "content": caption_example1},
                        {"role": "assistant", "content": "Category name: [car]"},
                        {"role": "user", "content": caption_example2},
                        {
                            "role": "assistant",
                            "content": "Category name: [traffic-sign]",
                        },
                        {"role": "user", "content": caption_example3},
                        {"role": "assistant", "content": "Category name: [terrain]"},
                        {"role": "user", "content": caption_example4},
                        {
                            "role": "assistant",
                            "content": "Category name: [traffic-sign]",
                        },
                        {"role": "user", "content": caption_obj},
                    ],
                ]
                # llama进行回答
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=None,
                    temperature=0.6,
                    top_p=0.9,
                )
                # 读取llama回答结果中的generation content作为caption融合结果
                for dialog, result in zip(dialogs, results):
                    input_text = result["generation"]["content"]
                    pattern = r"\[([^]]+)\]"  # 匹配方括号中的内容
                    match = re.search(pattern, input_text)
                    extracted_content = ""  # 빈 문자열로 초기화
                    if match:
                        extracted_content = match.group(1)
                    else:
                        # match가 없을 경우 sbert를 사용하여 처리
                        caption_ft = sbert_model.encode(
                            input_text, convert_to_tensor=True
                        )
                        caption_ft = caption_ft / caption_ft.norm(dim=-1, keepdim=True)
                        caption_ft = caption_ft.squeeze()
                        similarities = F.cosine_similarity(
                            class_name_fts, caption_ft.unsqueeze(0), dim=-1
                        )
                        max_indices = torch.argmax(similarities)
                        extracted_content = class_names_sk[max_indices]
                # 如果llama生成的特征没有在给定列表中，则使用sbert特征配准
                if extracted_content not in class_colors_sk_disk:
                    extracted_content_ft = sbert_model.encode(
                        extracted_content, convert_to_tensor=True
                    )
                    extracted_content_ft = (
                        extracted_content_ft
                        / extracted_content_ft.norm(dim=-1, keepdim=True)
                    )
                    extracted_content_ft = extracted_content_ft.squeeze()
                    # 与class计算相似性
                    similarities = F.cosine_similarity(
                        class_name_fts, extracted_content_ft.unsqueeze(0), dim=-1
                    )
                    max_indices = torch.argmax(similarities)
                    # 设置好类别和颜色
                    bg_objects[i]["class_sk"] = class_names_sk[max_indices]
                    bg_objects[i]["inst_color"] = class_colors_sk[max_indices]
                else:
                    bg_objects[i]["class_sk"] = extracted_content
                    bg_objects[i]["inst_color"] = (
                        np.array(class_colors_sk_disk[extracted_content]) / 255.0
                    )
    elif cfg.class_methods == "gpt":
        print("Asking gpt for class")
        openai.api_key = cfg.openai_key
        openai.api_base = cfg.api_base
        TIMEOUT = 25  # timeout in seconds
        DEFAULT_PROMPT = """
        You are a classifier that can categorize a caption phrase into one of the following categories based on a caption phrase.
        List of categories: [car, bicycle, motorcycle, truck, person, bicyclist, motorcyclist, road,
        parking, sidewalk, building, fence, vegetation, trunk, terrain, pole, traffic-sign]
        . You only need to generate one category name which must be included in this list. 
        The output format is 'Category name: [[your summarized category name itself]]'
        Note that I may enter all the captions at the same time, please output them in order
        Here's an example for you. 
        Input: 
        'a car parked on the street
        a red and white sign
        grass on the side of the road
        '. 
        You should output like this: 
        'Category name: [car]
        Category name: [traffic-sign]
        Category name: [terrain]
        '
        Emphasizing again: Do not provide words beyond the given list!!! 
        Please test it yourself and regenerate it if it exceeds the list.
        Emphasizing again: Do not provide words beyond the given list!!!
        """
        caption_objects = objects.get_stacked_str_torch("caption")
        batch_size = cfg.gpt_max_num
        num_batches = len(caption_objects) // batch_size + (
            len(caption_objects) % batch_size > 0
        )

        for batch_idx in range(num_batches):
            print("Batch num:", batch_idx, "/", num_batches)
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            current_caption_batch = caption_objects[start_idx:end_idx]
            caption_obj_batch = "\n".join(current_caption_batch)
            chat_completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": DEFAULT_PROMPT + "\n\n" + caption_obj_batch,
                    }
                ],
                timeout=TIMEOUT,  # Timeout in seconds
            )
            input_text_batch = chat_completion["choices"][0]["message"]["content"]
            input_text_batch = input_text_batch.split("\n")
            extracted_contents_batch = [
                re.search(r"\[([^]]+)\]", result).group(1)
                for result in input_text_batch
                if re.search(r"\[([^]]+)\]", result)
            ]
            for i in range(len(current_caption_batch)):
                # 如果gpt生成的特征没有在给定列表中，则使用sbert特征配准
                extracted_content = extracted_contents_batch[i]
                if extracted_content not in class_colors_sk_disk:
                    extracted_content_ft = sbert_model.encode(
                        extracted_content, convert_to_tensor=True
                    )
                    extracted_content_ft = (
                        extracted_content_ft
                        / extracted_content_ft.norm(dim=-1, keepdim=True)
                    )
                    extracted_content_ft = extracted_content_ft.squeeze()
                    # 与class计算相似性
                    similarities = F.cosine_similarity(
                        class_name_fts, extracted_content_ft.unsqueeze(0), dim=-1
                    )
                    max_indices = torch.argmax(similarities)
                    # 设置好类别和颜色
                    objects[start_idx + i]["class_sk"] = class_names_sk[max_indices]
                    objects[start_idx + i]["inst_color"] = class_colors_sk[max_indices]
                else:
                    objects[start_idx + i]["class_sk"] = extracted_content
                    objects[start_idx + i]["inst_color"] = (
                        np.array(class_colors_sk_disk[extracted_content]) / 255.0
                    )
        if bg_objects is not None:
            caption_obj = bg_objects.get_stacked_str_torch("caption")
            caption_obj = "\n".join(caption_obj)
            chat_completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": DEFAULT_PROMPT + "\n\n" + caption_obj}
                ],
                timeout=TIMEOUT,  # Timeout in seconds
            )
            input_text = chat_completion["choices"][0]["message"]["content"]
            input_text = input_text.split("\n")
            extracted_contents = [
                re.search(r"\[([^]]+)\]", result).group(1)
                for result in input_text
                if re.search(r"\[([^]]+)\]", result)
            ]
            for i in range(len(bg_objects)):
                bg_objects[i]["class_sk"] = extracted_contents[i]
                bg_objects[i]["inst_color"] = (
                    np.array(class_colors_sk_disk[extracted_contents[i]]) / 255.0
                )
    else:
        raise NotImplementedError
    return objects, bg_objects


def show_captions(objects: MapObjectList, bg_objects: MapObjectList):
    """
    展示objects所对应的caption以用于debug
    """
    for i in range(len(objects)):
        caption_obj = objects[i]["caption"]
        class_obj = objects[i]["class_sk"]
        print(f"object id {i} capitons: {caption_obj} ******** class_name: {class_obj}")
    if bg_objects is not None:
        for i in range(len(bg_objects)):
            caption_obj = bg_objects[i]["caption"]
            class_obj = bg_objects[i]["class_sk"]
            print(
                f"bgobject id {i} capitons: {caption_obj} ******** class_name: {class_obj}"
            )
