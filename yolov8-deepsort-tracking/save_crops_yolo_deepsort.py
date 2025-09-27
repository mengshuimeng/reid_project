# save_crops_yolo_deepsort.py
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import cv2  # opencv-python
from ultralytics import YOLO

import deep_sort.deep_sort.deep_sort as ds

def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """绘制带有背景的文本。"""
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

def extract_detections(results, detect_class):
    """
    从YOLOv8的results中提取检测框与置信度。
    注意：保持此函数与原来行为一致（避免改动你现在能跑通的流程）。
    返回:
      detections: numpy array (N,4)  —— 注意：格式与Tracker期望格式一致（你原本的实现可用）
      confarray: list of confidences
    """
    detections = np.empty((0, 4))
    confarray = []
    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                # 保持你原脚本的写法（可能返回xywh或者xyxy，视你的YOLO/DS对接而定）
                vals = box.xywh[0].int().tolist()  # 原脚本里这样写
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array(vals)))
                confarray.append(conf)
    return detections, confarray

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_label_map(map_dict: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(map_dict, f, indent=2, ensure_ascii=False)

def load_label_map(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def clamp(val, lo, hi):
    return max(lo, min(val, hi))

def crop_and_save(frame, x1, y1, x2, y2, out_path: Path, filename: str):
    h, w = frame.shape[:2]
    # clamp coords
    x1c = clamp(int(round(x1)), 0, w - 1)
    y1c = clamp(int(round(y1)), 0, h - 1)
    x2c = clamp(int(round(x2)), 0, w - 1)
    y2c = clamp(int(round(y2)), 0, h - 1)

    if x2c <= x1c or y2c <= y1c:
        # 不合法 bbox，跳过保存
        return False

    crop = frame[y1c:y2c, x1c:x2c]
    cv2.imwrite(str(out_path / filename), crop)
    return True

def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker,
                     camera_id: int = 1, sequence_id: int = 1,
                     save_crops: bool = True, label_map_path: str = None):
    """
    处理视频，检测并跟踪，同时将裁剪结果按命名规则保存：
      {person_label(4)}_c{camera}s{sequence}_{frame(6)}_{det_idx(2)}.jpg
    参数:
      camera_id, sequence_id: 摄像头与录像段编号（int），会形式化为 c1s1 嵌入文件名
      save_crops: 是否保存裁剪
      label_map_path: 可选，保存/加载 track_id -> label_no 的 JSON 文件路径，以便跨次运行保持 label 一致
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_dir(output_path)

    # crops目录放在 output_path/crops 下
    crops_root = output_path / "crops"
    ensure_dir(crops_root)

    # camera/sequence 子目录（便于管理）
    cam_seq_dir = crops_root / f"c{camera_id}s{sequence_id}"
    ensure_dir(cam_seq_dir)

    # label map: 把 tracker 的内部 ID 映射为连续的 label 编号（从1开始）
    label_map = {}
    if label_map_path:
        label_map_path = Path(label_map_path)
        if label_map_path.exists():
            label_map = load_label_map(label_map_path)
            # JSON 中的 keys 是字符串，转换回 int keys 较好，但为了简单，我们在内部使用 str(track_id)
    next_label = (max([int(v) for v in label_map.values()]) + 1) if label_map else 1

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video_path = output_path / "output.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, size, isColor=True)

    frame_idx = 0
    # 用于同一帧的检测计数（每帧重置）
    # label_map: track_id (string) -> label_no (int)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        det_idx_this_frame = 0  # 每帧检测框计数，从1开始

        # YOLO 预测
        results = model(frame, stream=True)

        # 提取检测框（保持原实现）
        detections, confarray = extract_detections(results, detect_class)

        # DeepSort 更新
        resultsTracker = tracker.update(detections, confarray, frame)

        # 如果 tracker 没有任何返回，仍然写入视频帧
        for trk in resultsTracker:
            # 请注意：不同的deep_sort实现返回格式不同，我们保持原始脚本对解包的兼容性
            # 期望 trk 可被解包为 (x1, y1, x2, y2, track_id)
            try:
                x1, y1, x2, y2, track_id = trk
            except Exception:
                # 如果返回的条目更长或不同，尝试取前5个元素
                x1, y1, x2, y2, track_id = trk[0], trk[1], trk[2], trk[3], trk[4]

            # 有些实现返回的是 cx,cy,w,h 而不是 x1,y1,x2,y2 —— 做个鲁棒判断
            x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)
            w_check = x2_f - x1_f
            h_check = y2_f - y1_f
            if w_check <= 0 or h_check <= 0:
                # 视为 (cx, cy, w, h)
                cx, cy, w_box, h_box = x1_f, y1_f, x2_f, y2_f
                x1c = cx - w_box / 2.0
                y1c = cy - h_box / 2.0
                x2c = cx + w_box / 2.0
                y2c = cy + h_box / 2.0
            else:
                x1c, y1c, x2c, y2c = x1_f, y1_f, x2_f, y2_f

            # 转为整数并裁剪范围
            x1i, y1i, x2i, y2i = int(round(x1c)), int(round(y1c)), int(round(x2c)), int(round(y2c))

            # tracker 内部 ID 可能是 int 或者字符串
            track_id_str = str(int(track_id)) if isinstance(track_id, (int, np.integer, float)) or (isinstance(track_id, (np.ndarray,)) and track_id.size==1) else str(track_id)

            # 分配/获取 label 编号（连续编号）
            if track_id_str not in label_map:
                label_map[track_id_str] = next_label
                next_label += 1
            label_no = int(label_map[track_id_str])

            # 保存裁剪图片（按命名规则）
            if save_crops:
                det_idx_this_frame += 1
                # 格式化命名
                label_str = str(label_no).zfill(4)        # 0001
                camera_str = f"c{camera_id}"             # c1
                sequence_str = f"s{sequence_id}"         # s1
                frame_str = str(frame_idx).zfill(6)      # 000151
                det_str = str(det_idx_this_frame).zfill(2)  # 01
                filename = f"{label_str}_{camera_str}{sequence_str}_{frame_str}_{det_str}.jpg"
                saved = crop_and_save(frame, x1i, y1i, x2i, y2i, cam_seq_dir, filename)
                if not saved:
                    # 若裁剪失败，可记录或打印
                    print(f"[WARN] skip save invalid crop: frame {frame_idx}, track {track_id_str}, bbox {(x1i,y1i,x2i,y2i)}")

            # 绘制 bbox 和 ID（保持原来的展示）
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (255, 0, 255), 3)
            putTextWithBackground(frame, str(label_no).zfill(4), (max(-10, x1i), max(40, y1i)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))

        # 写输出视频
        output_video.write(frame)

    output_video.release()
    cap.release()

    # 保存 label_map（如果指定了路径）
    if label_map_path:
        try:
            save_label_map(label_map, Path(label_map_path))
            print(f"label map saved to {label_map_path}")
        except Exception as e:
            print(f"failed to save label map: {e}")

    print(f'output video: {output_video_path}')
    print(f'crops saved to: {cam_seq_dir}')
    return output_video_path

if __name__ == "__main__":
    # —— 请按实际路径修改下面几项 —— #
    input_path = r"D:\code\python\Reid\yolov8-deepsort-tracking\VID20250906201700(3).mp4"
    output_path = r"D:\code\python\Reid\yolov8-deepsort-tracking\output"
    yolo_weights = "yolov8n.pt"  # 或你训练的权重
    deep_sort_ckpt = "deep_sort/deep_sort/deep/checkpoint/ckpt.t7"  # 保持你原来的路径
    camera_id = 1   # 设置为当前摄像头编号（1..6）
    sequence_id = 1 # 设置为当前录像段编号（1..N）
    label_map_json = str(Path(output_path) / "label_map_c1s1.json")  # 可选：保存track->label映射

    # 初始化模型与Tracker（与你的原始脚本相同）
    model = YOLO(yolo_weights)
    detect_class = 0  # person
    tracker = ds.DeepSort(deep_sort_ckpt)

    detect_and_track(input_path, output_path, detect_class, model, tracker,
                     camera_id=camera_id, sequence_id=sequence_id,
                     save_crops=True, label_map_path=label_map_json)
