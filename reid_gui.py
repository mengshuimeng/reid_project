# reid_gui.py
"""
桌面 GUI：用于上传视频、上传 query 图片、运行检测/跟踪并展示裁剪结果与匹配结果。
依赖：PyQt5, Pillow
运行：python reid_gui.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 保证路径正确
sys.path.append(os.path.join(os.getcwd(), 'yolov8-deepsort-tracking'))


import sys
import os
import subprocess
import threading
from pathlib import Path
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QScrollArea, QGridLayout,
    QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

from PIL import Image


# ---------- 配置（默认路径、脚本名） ----------
SAVE_CROPS_SCRIPT = "yolov8-deepsort-tracking/save_crops_yolo_deepsort.py"
FIND_MOST_SCRIPT = "find_most.py"

# 当使用 subprocess 调用脚本时的 python 可执行路径（可改）
PYTHON_BIN = sys.executable

# ---------- GUI 主类 ----------
class ReidGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReID 管线 GUI")
        self.resize(1100, 700)

        # 状态 / 参数
        self.video_path = ""
        self.query_image_path = ""
        self.output_root = Path.cwd() / "output"  # 默认输出根目录（save_crops 脚本输出）
        self.cam_id = 1
        self.seq_id = 1

        # UI 组件
        self.btn_select_video = QPushButton("上传视频（Select Video）")
        self.lbl_video = QLineEdit()
        self.lbl_video.setReadOnly(True)

        self.btn_select_query = QPushButton("上传待识别图片（Select Query Image）")
        self.lbl_query = QLineEdit()
        self.lbl_query.setReadOnly(True)

        self.btn_run_detection = QPushButton("运行检测并保存裁剪（Run Detection）")
        self.btn_run_matching = QPushButton("运行匹配（Run Matching）")

        self.btn_open_output_folder = QPushButton("打开输出文件夹")
        self.btn_refresh_thumbs = QPushButton("刷新输出缩略图")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        self.thumb_area = QScrollArea()
        self.thumb_widget = QWidget()
        self.thumb_layout = QGridLayout()
        self.thumb_widget.setLayout(self.thumb_layout)
        self.thumb_area.setWidgetResizable(True)
        self.thumb_area.setWidget(self.thumb_widget)

        # 摄像机/片段输入
        self.cam_input = QLineEdit(str(self.cam_id))
        self.seq_input = QLineEdit(str(self.seq_id))

        # 布局
        self._build_layout()
        self._connect_signals()

        self.log("GUI ready. Put scripts in same folder as this GUI or ensure importability.")

    def _build_layout(self):
        row1 = QHBoxLayout()
        row1.addWidget(self.btn_select_video)
        row1.addWidget(self.lbl_video)
        row1.addWidget(QLabel("Camera ID"))
        row1.addWidget(self.cam_input)
        row1.addWidget(QLabel("Sequence ID"))
        row1.addWidget(self.seq_input)

        row2 = QHBoxLayout()
        row2.addWidget(self.btn_select_query)
        row2.addWidget(self.lbl_query)
        row2.addWidget(self.btn_run_detection)
        row2.addWidget(self.btn_run_matching)

        row3 = QHBoxLayout()
        row3.addWidget(self.btn_open_output_folder)
        row3.addWidget(self.btn_refresh_thumbs)

        left_layout = QVBoxLayout()
        left_layout.addLayout(row1)
        left_layout.addLayout(row2)
        left_layout.addLayout(row3)
        left_layout.addWidget(QLabel("日志 / 进度 Log"))
        left_layout.addWidget(self.log_text)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addWidget(self.thumb_area, 3)

        self.setLayout(main_layout)

    def _connect_signals(self):
        self.btn_select_video.clicked.connect(self.select_video)
        self.btn_select_query.clicked.connect(self.select_query)
        self.btn_run_detection.clicked.connect(self.run_detection_clicked)
        self.btn_run_matching.clicked.connect(self.run_matching_clicked)
        self.btn_open_output_folder.clicked.connect(self.open_output_folder)
        self.btn_refresh_thumbs.clicked.connect(self.refresh_thumbnails)

    # ---------- 日志 ----------
    def log(self, text):
        self.log_text.append(text)
        print(text)

    # ---------- 交互动作 ----------
    def select_video(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择视频文件", str(Path.cwd()), "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if fp:
            self.video_path = fp
            self.lbl_video.setText(fp)
            self.log(f"selected video: {fp}")

    def select_query(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择查询图片", str(Path.cwd()), "Image Files (*.jpg *.png *.jpeg);;All Files (*)")
        if fp:
            self.query_image_path = fp
            self.lbl_query.setText(fp)
            self.log(f"selected query image: {fp}")

    def run_detection_clicked(self):
        if not self.video_path:
            QMessageBox.warning(self, "缺少视频", "请先选择视频后再运行检测。")
            return
        # 读取 cam/seq
        try:
            self.cam_id = int(self.cam_input.text())
            self.seq_id = int(self.seq_input.text())
        except:
            QMessageBox.warning(self, "参数错误", "Camera ID / Sequence ID 必须为整数。")
            return

        # 在单独线程运行检测（避免阻塞 UI）
        t = threading.Thread(target=self._run_detection_thread, daemon=True)
        t.start()

    def run_matching_clicked(self):
        if not self.query_image_path:
            QMessageBox.warning(self, "缺少查询图片", "请先选择查询图片再运行匹配。")
            return
        t = threading.Thread(target=self._run_matching_thread, daemon=True)
        t.start()

    def open_output_folder(self):
        out = str(self.output_root.resolve())
        if not Path(out).exists():
            QMessageBox.information(self, "提示", f"输出文件夹不存在：{out}")
            return
        if sys.platform.startswith("win"):
            os.startfile(out)
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", out])
        else:
            subprocess.Popen(["xdg-open", out])

    # ---------- 核心运行逻辑（尝试 import，否则 subprocess） ----------
    def _run_detection_thread(self):
        self.log("开始运行检测 (detect_and_track)...")
        # 尝试直接 import save_crops_yolo_deepsort.detect_and_track
        try:
            import importlib
            spec_mod = importlib.import_module("save_crops_yolo_deepsort")
            if hasattr(spec_mod, "detect_and_track"):
                self.log("模块导入成功：save_crops_yolo_deepsort.detect_and_track，开始调用...")
                # prepare args
                output_path = str(self.output_root)
                model_weights = getattr(spec_mod, "yolo_weights", None)  # optional
                # call function - many variants: detect_and_track(input_path, output_path, detect_class, model, tracker, camera_id, sequence_id, save_crops=True, label_map_path=None)
                # 你的脚本 defines model loading inside __main__; our GUI will try to call detect_and_track in a minimal way.
                # We'll look for a wrapper function `run_from_paths` else we try to call detect_and_track with the minimal set.
                if hasattr(spec_mod, "run_from_paths"):
                    # 如果脚本提供了方便的包装函数 run_from_paths(input_path, output_path, camera_id, sequence_id)
                    try:
                        spec_mod.run_from_paths(self.video_path, output_path, int(self.cam_id), int(self.seq_id))
                        self.log("run_from_paths 调用完成。")
                    except Exception as e:
                        self.log(f"调用 run_from_paths 出错: {e}")
                else:
                    # fallback: call detect_and_track in a subprocess because detect_and_track expects model/tracker objects
                    self.log("未找到 run_from_paths；使用 subprocess 调用脚本（推荐方式）。")
                    self._run_detection_subprocess()
            else:
                self.log("模块导入成功但未找到 detect_and_track，使用 subprocess。")
                self._run_detection_subprocess()
        except Exception as e:
            self.log(f"尝试 import save_crops 模块失败：{e}，改用 subprocess 调用脚本。")
            self._run_detection_subprocess()

        # 刷新缩略图
        self.refresh_thumbnails()

    def _run_detection_subprocess(self):
        # 以 subprocess 方式调用外部脚本，传递参数（video, output_dir, camera_id, sequence_id）
        cmd = [
            PYTHON_BIN, SAVE_CROPS_SCRIPT,
            "--input", self.video_path,
            "--output", str(self.output_root),
            "--camera_id", str(self.cam_id),
            "--sequence_id", str(self.seq_id)
        ]
        self.log("subprocess cmd: " + " ".join(cmd))
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in p.stdout:
                self.log(line.rstrip())
            p.wait()
            self.log(f"检测脚本退出，返回码 {p.returncode}")
        except Exception as e:
            self.log(f"运行脚本失败: {e}")

    def _run_matching_thread(self):
        self.log("开始运行匹配 (find_most)...")
        # 尝试直接 import find_most.run_find_most
        try:
            import importlib
            fm = importlib.import_module("find_most")
            if hasattr(fm, "run_find_most"):
                self.log("模块导入成功：find_most.run_find_most，开始调用...")
                # 假设 run_find_most(query_path, crops_folder, output_folder)
                crops_folder = str(self.output_root / "crops" / f"c{self.cam_id}s{self.seq_id}")
                out_folder = str(self.output_root / "matched")
                Path(out_folder).mkdir(parents=True, exist_ok=True)
                try:
                    fm.run_find_most(self.query_image_path, crops_folder, out_folder)
                    self.log("find_most.run_find_most 执行完成。")
                except Exception as e:
                    self.log(f"调用 run_find_most 出错: {e}")
            else:
                self.log("未找到 run_find_most，使用 subprocess 调用脚本。")
                self._run_matching_subprocess()
        except Exception as e:
            self.log(f"尝试 import find_most 模块失败：{e}，改用 subprocess 调用脚本。")
            self._run_matching_subprocess()

        # 刷新缩略图
        self.refresh_thumbnails()

    def _run_matching_subprocess(self):
        crops_folder = str(self.output_root / "crops" / f"c{self.cam_id}s{self.seq_id}")
        out_folder = str(self.output_root / "matched")
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        # 调用 find_most.py，假定它接受命令行参数: --query, --crops, --out
        cmd = [
            PYTHON_BIN, FIND_MOST_SCRIPT,
            "--query", self.query_image_path,
            "--crops", crops_folder,
            "--out", out_folder
        ]
        self.log("subprocess cmd: " + " ".join(cmd))
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in p.stdout:
                self.log(line.rstrip())
            p.wait()
            self.log(f"匹配脚本退出，返回码 {p.returncode}")
        except Exception as e:
            self.log(f"运行匹配脚本失败: {e}")

    # ---------- 缩略图展示 ----------
    def refresh_thumbnails(self):
        # 清空布局
        for i in reversed(range(self.thumb_layout.count())):
            w = self.thumb_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        # 输出目录：默认 output/crops/c{cam} s{seq}
        crops_dir = Path(self.output_root) / "crops" / f"c{self.cam_id}s{self.seq_id}"
        if not crops_dir.exists():
            self.log(f"未找到裁剪目录: {crops_dir}")
            return

        img_paths = sorted([p for p in crops_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not img_paths:
            self.log(f"裁剪目录中没有图片: {crops_dir}")
            return

        # 每行多少列
        cols = 3
        thumb_size = 200

        row = 0
        col = 0
        for p in img_paths:
            try:
                pil = Image.open(p)
                pil.thumbnail((thumb_size, thumb_size))
                data = pil.convert("RGBA").tobytes("raw", "RGBA")
                qimg = QImage(data, pil.width, pil.height, QImage.Format_RGBA8888)
                pix = QPixmap.fromImage(qimg)
                lbl = QLabel()
                lbl.setPixmap(pix)
                lbl.setScaledContents(True)
                lbl.setFixedSize(QSize(thumb_size, thumb_size))
                lbl.setToolTip(str(p.name))
                lbl.mousePressEvent = partial(self._on_thumb_click, str(p))
                self.thumb_layout.addWidget(lbl, row, col)
                col += 1
                if col >= cols:
                    col = 0
                    row += 1
            except Exception as e:
                self.log(f"加载缩略图失败 {p}: {e}")

        self.log(f"已加载 {len(img_paths)} 张缩略图。")

    def _on_thumb_click(self, img_path, event):
        # 显示大图在新窗口
        try:
            dlg = ImagePreviewDialog(img_path)
            dlg.exec_()
        except Exception as e:
            self.log(f"打开图片预览失败: {e}")

# ---------- 大图预览对话框 ----------
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
class ImagePreviewDialog(QDialog):
    def __init__(self, path):
        super().__init__()
        self.setWindowTitle(Path(path).name)
        self.resize(800, 800)
        v = QVBoxLayout()
        lbl = QLabel()
        pix = QPixmap(path)
        lbl.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        v.addWidget(lbl)
        btn_open = QPushButton("在文件管理器中打开此文件")
        btn_open.clicked.connect(lambda: os.startfile(path) if sys.platform.startswith("win") else subprocess.Popen(["xdg-open", path]))
        v.addWidget(btn_open)
        self.setLayout(v)

# ---------- 启动 ----------
def main():
    app = QApplication(sys.argv)
    gui = ReidGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
