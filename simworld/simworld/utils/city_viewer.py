# simworld/utils/city_viewer.py
# -*- coding: utf-8 -*-
import json
from typing import List, Tuple, Callable

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsRectItem, QTextEdit
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import pyqtgraph as pg

# 依赖：SimWorld（调用方需先把 SimWorld 路径加到 sys.path）
from simworld.map.map import Map
from simworld.utils.vector import Vector

# world.json 中的建筑宽高单位（米）到渲染坐标（厘米）的比例
WORLD_SCALE = 100.0


# ------------------------- 工具函数 ------------------------- #
def thin_by_min_spacing_xy(points_xy: List[Tuple[float, float]], min_dist_cm: float) -> List[Tuple[float, float]]:
    """按最小间距抽稀坐标序列（单位：cm）。"""
    if not points_xy:
        return []
    out = [points_xy[0]]
    last_x, last_y = points_xy[0]
    min2 = float(min_dist_cm) * float(min_dist_cm)
    for x, y in points_xy[1:]:
        dx = x - last_x
        dy = y - last_y
        if dx*dx + dy*dy >= min2:
            out.append((x, y))
            last_x, last_y = x, y
    if out[-1] != points_xy[-1]:
        out.append(points_xy[-1])
    return out


def build_city_map(CONFIG: dict) -> Map:
    """
    依据 roads.json 构建带细分节点的 Map。
    需要 CONFIG 包含：
      - "map.input_roads"
      - "traffic.sidewalk_offset"
    """
    m = Map(config=CONFIG)
    m.initialize_map_from_file(
        roads_file=CONFIG["map.input_roads"],
        sidewalk_offset=CONFIG["traffic.sidewalk_offset"],
        fine_grained=True,
        num_waypoints_normal=3,
        waypoints_distance=1500,       # 沿边每 15m 生成插值点（单位 cm）
        waypoints_normal_distance=150  # 法向多层点间距 1.5m
    )
    return m


# ------------------------- 视图类 ------------------------- #
class CityViewer(QMainWindow):
    """
    通用城市可视化：
      - 背景：建筑 + 公交路线 +（本文件补充）公交站台 / 充电桩
      - 叠加：任意路线（plot_route）、散点（plot_points）
      - 信息面板：set_info_text
      - 可添加按钮：add_button
    """
    def __init__(
        self,
        world_json_path: str,
        building_defs_path: str,
        title: str = "City Viewer",
        background_color: str = "#F7FBFF",
        show_grid: bool = True,
    ):
        super().__init__()
        self.setWindowTitle(title)

        # 读取 world.json（包含场景节点与公交线路）
        with open(world_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._nodes = data.get("nodes", [])
        self._bus_routes = data.get("bus_routes", [])

        # building defs（从 UE 导出的建筑蓝图宽/高等）
        with open(building_defs_path, "r", encoding="utf-8") as f:
            defs = json.load(f)
        self._building_defs = {}
        for b in defs.get("buildings", []):
            btype = b.get("type")
            bounds = b.get("bounds", {}) or {}
            if btype and "width" in bounds and "height" in bounds:
                self._building_defs[btype] = {
                    "width": float(bounds["width"]),
                    "height": float(bounds["height"])
                }

        # 预取建筑中心（供测试时随机抽）
        self._buildings_xy: List[Tuple[float, float]] = []
        for n in self._nodes:
            if str(n.get("instance_name", "")).startswith("BP_Building"):
                loc = (n.get("properties", {}) or {}).get("location", {}) or {}
                x = float(loc.get("x", 0.0))
                y = float(loc.get("y", 0.0))
                self._buildings_xy.append((x, y))

        # ---------------- UI ----------------
        self._main = QWidget()
        self.setCentralWidget(self._main)
        v = QVBoxLayout(self._main)

        # 顶部标题 + 动作按钮区
        self._header = QHBoxLayout()
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignCenter)
        self._header.addWidget(self._title_label)
        v.addLayout(self._header)

        # 绘图区
        self.plot = pg.PlotWidget()
        self.plot.setBackground(background_color)
        self.plot.setAspectLocked(True)
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.setMenuEnabled(False)
        if show_grid:
            self.plot.showGrid(True, True, alpha=0.25)
        v.addWidget(self.plot)

        # 信息区（默认高度 150，可通过 API 调整）
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(150)
        v.addWidget(self.info)

        # 底图
        self._draw_base()

    # ---------------- 公共 API ----------------
    def add_button(self, text: str, on_click: Callable[[], None]) -> QPushButton:
        btn = QPushButton(text)
        btn.clicked.connect(on_click)
        self._header.addWidget(btn)
        return btn

    def set_info_text(self, text: str):
        self.info.setText(text)

    def set_info_area_height(self, height_px: int):
        """调整信息区高度（像素）。"""
        self.info.setFixedHeight(int(height_px))

    def set_info_font_point_size(self, pt: int):
        """调整信息区字体大小（point）。"""
        f: QFont = self.info.font()
        f.setPointSize(int(pt))
        self.info.setFont(f)

    def clear_overlays(self):
        """清除叠加层并重画底图。"""
        self.plot.clear()
        self._draw_base()

    def plot_route(
        self,
        points_xy: List[Tuple[float, float]],
        color: str = "#FF5722",
        width: float = 4.0,
        scatter: bool = False,
        scatter_size: float = 8.0,
        show_endpoints: bool = True,
    ):
        """绘制折线（可选散点）并强调起终点。"""
        if len(points_xy) >= 2:
            xs = [p[0] for p in points_xy]
            ys = [p[1] for p in points_xy]
            self.plot.addItem(pg.PlotDataItem(
                x=xs, y=ys, pen=pg.mkPen(color, width=width), antialias=True
            ))
        if scatter and points_xy:
            self.plot.addItem(pg.ScatterPlotItem(
                pos=points_xy, size=scatter_size,
                pen=pg.mkPen("w", width=1), brush=pg.mkBrush(color),
                symbol="o", antialias=True
            ))
        if show_endpoints and points_xy:
            sx, sy = points_xy[0]
            ex, ey = points_xy[-1]
            self.plot.addItem(pg.ScatterPlotItem(
                pos=[(sx, sy)], size=14, pen=pg.mkPen("w", width=2),
                brush=pg.mkBrush("#2ECC71"), symbol="o", antialias=True
            ))
            self.plot.addItem(pg.ScatterPlotItem(
                pos=[(ex, ey)], size=14, pen=pg.mkPen("w", width=2),
                brush=pg.mkBrush("#E74C3C"), symbol="o", antialias=True
            ))

    def plot_points(
        self,
        points_xy: List[Tuple[float, float]],
        size: float = 12.0,
        color: str = "#00BCD4",
        symbol: str = "o",
    ):
        """绘制一组散点。"""
        if not points_xy:
            return
        self.plot.addItem(pg.ScatterPlotItem(
            pos=points_xy, size=size,
            pen=pg.mkPen("w", width=1), brush=pg.mkBrush(color),
            symbol=symbol, antialias=True
        ))

    def get_buildings_xy(self) -> List[Tuple[float, float]]:
        """返回建筑中心点坐标列表（单位：cm）。"""
        return list(self._buildings_xy)

    def save_png(self, out_path: str) -> bool:
        """保存当前视图为 PNG。"""
        pixmap = self.plot.grab()
        return bool(pixmap.save(out_path, "PNG"))

    # ---------------- 内部：底图 ----------------
    def _draw_base(self):
        """绘制建筑 + 公交路线 +（POI）公交站台/充电桩。"""
        # 建筑（矩形，旋转）
        for node in self._nodes:
            inst = node.get("instance_name", "")
            props = node.get("properties", {}) or {}
            loc = props.get("location", {}) or {}
            ori = props.get("orientation", {}) or {}
            x = float(loc.get("x", 0.0))
            y = float(loc.get("y", 0.0))
            yaw = float(ori.get("yaw", 0.0))

            if inst.startswith("BP_Building"):
                size = self._building_defs.get(inst, {"width": 1.0, "height": 1.0})
                w = float(size["width"]) * WORLD_SCALE
                h = float(size["height"]) * WORLD_SCALE
                rect = QGraphicsRectItem(x - w / 2.0, y - h / 2.0, w, h)
                rect.setTransformOriginPoint(x, y)
                rect.setRotation(yaw)
                rect.setPen(pg.mkPen("k", width=1))
                rect.setBrush(pg.mkBrush("#7F8C8D"))
                self.plot.addItem(rect)

        # 公交路线（world.json 的 path 坐标通常为米，这里乘 WORLD_SCALE → cm）
        for route in self._bus_routes:
            path = route.get("path", [])
            if len(path) >= 2:
                xs = [float(p.get("x", 0.0)) * WORLD_SCALE for p in path]
                ys = [float(p.get("y", 0.0)) * WORLD_SCALE for p in path]
                self.plot.addItem(pg.PlotDataItem(
                    x=xs, y=ys, pen=pg.mkPen("#FFA500", width=2.0), antialias=True
                ))

        # POIs：公交站台 & 充电桩（直接使用 world.json 中的坐标）
        for node in self._nodes:
            inst = node.get("instance_name", "")
            props = node.get("properties", {}) or {}
            loc = props.get("location", {}) or {}
            x = float(loc.get("x", 0.0))
            y = float(loc.get("y", 0.0))
            poi_type = props.get("poi_type") or props.get("type")

            # 充电桩
            if poi_type == "charging_station" or inst.startswith("POI_ChargingStation"):
                self.plot.addItem(pg.ScatterPlotItem(
                    pos=[(x, y)], size=12,
                    pen=pg.mkPen("w", width=1),
                    brush=pg.mkBrush("#27AE60"),  # 绿色
                    symbol="o", antialias=True
                ))
            # 公交站
            elif poi_type == "bus_station" or inst.startswith("POI_BusStation"):
                self.plot.addItem(pg.ScatterPlotItem(
                    pos=[(x, y)], size=14,
                    pen=pg.mkPen("#8E5A0A", width=1),
                    brush=pg.mkBrush("#F39C12"),  # 橙色
                    symbol="t", antialias=True
                ))
