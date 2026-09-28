#!/usr/bin/env python3
"""按 Go2_PIE / terrain.make_terrain 公式生成跑酷 MuJoCo XML。

每种地形一个文件；文件内沿 +X 依次铺 num_rows 段难度（对齐 Isaac Gym curriculum）:
  difficulty = row / num_rows,  row = 0..num_rows-1
  即默认 10 段: 0.0, 0.1, ..., 0.9（易 → 难），每段 LENGTH=10m。

索引 0..10（对齐训练有效跑酷类 + 平地）:
  0 gap  1 stones  2 stones2  3 stones1  4 bridge  5 hurdle
  6 ramp  7 half_stairs_025  8 half_stairs_030  9 rough  10 flat

用法（仓库根目录）:
  python deploy_mujoco/generate_parkour_mjcf.py
  python deploy_mujoco/generate_parkour_mjcf.py --num-rows 10
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

# 与 Go2_PIE_Config.terrain / make_terrain 对齐
HS = 0.05          # horizontal_scale [m]
VS = 0.005         # vertical_scale [m]
LENGTH = 10.0      # terrain_length [m] 前进方向 +X（单段）
WIDTH = 4.0        # terrain_width [m]  横向 Y∈[-W/2, W/2]
DEPTH = 0.6        # 坑深 [m]
PLAT = 2.0         # 入口平台长度 [m]
PLAT_HALF_Z = 0.05 # 平台半厚，顶面 z=0


@dataclass
class Box:
    name: str
    x: float
    y: float
    z: float
    hx: float
    hy: float
    hz: float
    rgba: Optional[str] = None


@dataclass
class Scene:
    boxes: List[Box] = field(default_factory=list)
    floor_z: float = -(DEPTH + 0.05)

    def add(self, name, x, y, z, hx, hy, hz, rgba=None):
        if hx < 1e-4 or hy < 1e-4 or hz < 1e-4:
            return
        self.boxes.append(Box(name, x, y, z, hx, hy, hz, rgba))

    def offset_x(self, dx: float, name_prefix: str = "") -> "Scene":
        out = Scene(floor_z=self.floor_z)
        for b in self.boxes:
            out.boxes.append(
                Box(
                    name=f"{name_prefix}{b.name}",
                    x=b.x + dx,
                    y=b.y,
                    z=b.z,
                    hx=b.hx,
                    hy=b.hy,
                    hz=b.hz,
                    rgba=b.rgba,
                )
            )
        return out

    def extend(self, other: "Scene"):
        self.floor_z = min(self.floor_z, other.floor_z)
        self.boxes.extend(other.boxes)


def _hash01(i: int, j: int = 0) -> float:
    """确定性 [0,1)，替代训练里的 np.random。"""
    return (math.sin(i * 12.9898 + j * 78.233) * 43758.5453) % 1.0


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _step_height(difficulty: float) -> float:
    # 连续：0.05 → 0.23（不再因 <0.2 平台期导致相邻 lv 一样）
    return 0.05 + 0.18 * difficulty


def _stone_size_grid(difficulty: float) -> float:
    # 训练式 -0.5*d^2+0.7，但去掉 <0.2 常数段 → 石块随难度变小
    return _clip(-0.5 * difficulty * difficulty + 0.7, 0.25, 0.7)


def _stone_size_row(difficulty: float) -> float:
    return _clip(-0.5 * difficulty + 0.8, 0.30, 0.8)


def _stone_dist(difficulty: float, easy: float = 0.05) -> float:
    # 连续拉大间距：easy → 0.4
    return easy + (0.4 - easy) * difficulty


def _bridge_size(difficulty: float) -> float:
    # 连续变窄：0.6 → 0.2
    return _clip(-0.6 * difficulty + 0.8, 0.20, 0.8)


def _gap_size(difficulty: float) -> float:
    """断崖宽度 [m]。训练用 cell clip 会把多档压成同一值；这里连续 0.15→1.0。"""
    return 0.15 + 0.85 * difficulty


def _add_platform_top(
    scene: Scene, name: str, x0: float, x1: float, y0: float, y1: float, top_z: float = 0.0
):
    bot = -DEPTH
    hz = 0.5 * (top_z - bot)
    if hz < 1e-4:
        return
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (top_z + bot)
    scene.add(name, cx, cy, cz, 0.5 * (x1 - x0), 0.5 * (y1 - y0), hz)


def _add_ground_slab(scene: Scene, name: str, x0: float, x1: float, y0: float, y1: float):
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    scene.add(name, cx, cy, -PLAT_HALF_Z, 0.5 * abs(x1 - x0), 0.5 * abs(y1 - y0), PLAT_HALF_Z)


# ---------------------------------------------------------------------------
# 各地形公式（几何语义对齐 add_extreme_gap_terrain）
# ---------------------------------------------------------------------------

def build_gap(difficulty: float) -> Scene:
    scene = Scene()
    gap = _gap_size(difficulty)
    y0, y1 = -WIDTH * 0.5, WIDTH * 0.5
    _add_ground_slab(scene, "plat_entry", 0.0, PLAT, y0, y1)
    x_a0, x_a1 = PLAT, LENGTH * 0.5
    # 保证岛还有可站宽度
    ix0 = min(x_a0 + gap, x_a1 - 0.4)
    ix1 = max(x_a1 - gap, ix0 + 0.4)
    iy0 = min(y0 + gap, y1 - 0.4)
    iy1 = max(y1 - gap, iy0 + 0.4)
    _add_ground_slab(scene, "island0", ix0, ix1, iy0, iy1)
    x_b0 = LENGTH * 0.5 + PLAT * 0.5
    x_b1 = LENGTH
    jx0 = min(x_b0 + gap, x_b1 - 0.4)
    jx1 = max(x_b1 - gap, jx0 + 0.4)
    _add_ground_slab(scene, "island1", jx0, jx1, iy0, iy1)
    _add_ground_slab(scene, "plat_exit", LENGTH - 1.0, LENGTH, -0.5, 0.5)
    return scene


def build_stones(difficulty: float) -> Scene:
    scene = Scene()
    stone = _stone_size_grid(difficulty)
    dist = _stone_dist(difficulty, easy=0.05)
    h_max = _step_height(difficulty)  # 0.05→0.23 连续
    y0, y1 = -WIDTH * 0.5 + 0.1, WIDTH * 0.5 - 0.1
    _add_ground_slab(scene, "plat_entry", 0.0, PLAT, -PLAT * 0.5, PLAT * 0.5)

    x = PLAT
    col = 0
    while x < LENGTH - 0.5:
        x1 = min(LENGTH - 0.3, x + stone)
        y = y0 + _hash01(col, 1) * min(stone, 0.4)
        row = 0
        if y > y0 + 0.05:
            top = 0.02 + h_max * (0.3 + 0.7 * _hash01(col, 0))
            _add_platform_top(scene, f"s_{col}_{row}", x, x1, y0, y, top)
            row += 1
        yy = y
        while yy < y1 - 0.1:
            y2 = min(y1, yy + stone)
            top = 0.02 + h_max * (0.3 + 0.7 * _hash01(col, row))
            _add_platform_top(scene, f"s_{col}_{row}", x, x1, yy, y2, top)
            yy = y2 + dist
            row += 1
        x = x1 + dist
        col += 1
    _add_ground_slab(scene, "plat_entry2", 0.0, PLAT, -PLAT * 0.5, PLAT * 0.5)
    _add_ground_slab(scene, "plat_exit", LENGTH - 1.0, LENGTH, -PLAT * 0.5, PLAT * 0.5)
    return scene


def build_stones2(difficulty: float) -> Scene:
    scene = Scene()
    stone = _clip(_stone_size_row(difficulty), 0.30, 0.80)
    dist = _stone_dist(difficulty, easy=0.08)
    h_max = _step_height(difficulty)
    plat_y0, plat_y1 = -PLAT * 0.5, PLAT * 0.5
    row1_y = -stone - 0.08
    row2_y = 0.08
    _add_ground_slab(scene, "plat_entry", 0.0, PLAT, plat_y0, plat_y1)

    x = PLAT
    i = 0
    while x < LENGTH - 0.5:
        x1 = min(LENGTH - 0.3, x + stone)
        for r, yb in enumerate((row1_y, row2_y)):
            top = 0.02 + h_max * (0.3 + 0.7 * _hash01(i, r))
            _add_platform_top(scene, f"t_{i}_{r}", x, x1, yb, yb + stone, top)
        x = x1 + dist
        i += 1
    _add_ground_slab(scene, "plat_entry2", 0.0, PLAT, plat_y0, plat_y1)
    _add_ground_slab(scene, "plat_exit", LENGTH - 1.0, LENGTH, plat_y0, plat_y1)
    return scene


def build_stones1(difficulty: float) -> Scene:
    scene = Scene()
    stone = _stone_size_row(difficulty)
    step_h = _step_height(difficulty)
    # 高度上限随难度抬；避开 VS*20=0.1 的硬顶导致后半程不再变难
    h_max = _clip(step_h, 0.02, 0.25)
    # 石间距连续：0.2 → 0.8
    gap = 0.20 + 0.60 * difficulty
    plat_y0, plat_y1 = -PLAT * 0.5, PLAT * 0.5
    _add_ground_slab(scene, "plat_entry", 0.0, PLAT, plat_y0, plat_y1)

    x = PLAT
    i = 0
    while x < LENGTH - 0.5:
        sx = _clip(stone, 0.35, 0.8)
        # 横向宽度随难度变窄
        sy = 0.90 - 0.50 * difficulty + 0.05 * (_hash01(i, 2) - 0.5)
        row_y = -0.5 * sy
        x1 = min(LENGTH - 0.3, x + sx)
        top = 0.02 + h_max * (0.4 + 0.6 * _hash01(i, 3))
        _add_platform_top(scene, f"o_{i}", x, x1, row_y, row_y + sy, top)
        x = x1 + gap
        i += 1
    _add_ground_slab(scene, "plat_entry2", 0.0, PLAT, plat_y0, plat_y1)
    _add_ground_slab(scene, "plat_exit", LENGTH - 1.0, LENGTH, plat_y0, plat_y1)
    return scene


def build_bridge(difficulty: float) -> Scene:
    scene = Scene()
    stone = _bridge_size(difficulty)
    h_amp = 0.01 + 0.08 * difficulty
    plat_y0, plat_y1 = -PLAT * 0.5, PLAT * 0.5
    row_y = -0.5 * stone
    _add_ground_slab(scene, "plat_entry", 0.0, PLAT, plat_y0, plat_y1)

    x = PLAT
    i = 0
    while x < LENGTH - 0.5:
        x1 = min(LENGTH - 0.3, x + stone)
        top = h_amp * _hash01(i, 4)
        _add_platform_top(scene, f"b_{i}", x, x1, row_y, row_y + stone, top)
        x = x1
        i += 1
    _add_ground_slab(scene, "plat_entry2", 0.0, PLAT, plat_y0, plat_y1)
    _add_ground_slab(scene, "plat_exit", LENGTH - 1.0, LENGTH, plat_y0, plat_y1)
    return scene


def build_hurdle(difficulty: float) -> Scene:
    scene = Scene()
    # 栏高连续：0.08 → 0.55；训练里 clip(diff/hs,60,80) 会让间距永远 3m
    hurdle_h = 0.08 + 0.47 * difficulty
    # 栏间距连续变密一点但仍可跑：3.5 → 2.0
    gap = 3.5 - 1.5 * difficulty
    sx = 0.45 - 0.15 * difficulty
    sy = 0.90 - 0.20 * difficulty
    corridor_w = 18 * HS
    _add_ground_slab(scene, "corridor", 0.0, LENGTH, -corridor_w * 0.5, corridor_w * 0.5)

    x = PLAT + 0.8
    i = 0
    while x < LENGTH - 1.0:
        x1 = x + sx
        hz = 0.5 * hurdle_h
        scene.add(
            f"hurdle_{i}",
            0.5 * (x + x1),
            0.0,
            hz,
            0.5 * sx,
            0.5 * sy,
            hz,
            rgba="0.8 0.35 0.2 1",
        )
        x = x1 + gap
        i += 1
    return scene


def build_ramp(difficulty: float) -> Scene:
    scene = Scene()
    min_s = 2.0
    max_s = math.tan(math.radians(33.0)) * HS / VS
    strength = min_s + (max_s - min_s) * difficulty
    y0, y1 = -WIDTH * 0.5 + 20 * HS, WIDTH * 0.5 - 20 * HS
    p = PLAT
    up_cells = int(p / HS)
    max_h = strength * up_cells * VS

    _add_ground_slab(scene, "entry", 0.0, p, y0, y1)
    n = max(4, up_cells // 2)
    for k in range(n):
        t0, t1 = k / n, (k + 1) / n
        x0, x1 = p + t0 * p, p + t1 * p
        top = strength * (t1 * up_cells) * VS
        _add_platform_top(scene, f"up_{k}", x0, x1, y0, y1, top)
    _add_platform_top(scene, "mid", 2 * p, 3 * p, y0, y1, max_h)
    for k in range(n):
        t0, t1 = k / n, (k + 1) / n
        x0, x1 = 3 * p + t0 * p, 3 * p + t1 * p
        top = max(0.0, max_h - strength * (t1 * up_cells) * VS)
        _add_platform_top(scene, f"down_{k}", x0, x1, y0, y1, top)
    _add_ground_slab(scene, "exit", 4 * p, LENGTH, y0, y1)
    return scene


def build_half_stairs(difficulty: float, step_width: float) -> Scene:
    scene = Scene()
    step_h = _step_height(difficulty)
    y0, y1 = -WIDTH * 0.5 + 20 * HS, WIDTH * 0.5 - 20 * HS
    p = PLAT
    _add_ground_slab(scene, "entry", 0.0, p, y0, y1)

    x = p
    h = 0.0
    k = 0
    while x < 2 * p - 1e-9:
        h += step_h
        x1 = min(x + step_width, 2 * p)
        _add_platform_top(scene, f"up_{k}", x, x1, y0, y1, h)
        x = x1
        k += 1
    max_h = h
    _add_platform_top(scene, "mid", 2 * p, 3 * p, y0, y1, max_h)

    x = 3 * p
    h = max_h
    k = 0
    while x < min(3 * p + p, LENGTH) and h > 0:
        x1 = min(x + step_width, 3 * p + p)
        _add_platform_top(scene, f"down_{k}", x, x1, y0, y1, h)
        h = max(0.0, h - step_h)
        x = x1
        k += 1
    _add_ground_slab(scene, "exit", max(x, 4 * p), LENGTH, y0, y1)
    return scene


def build_rough(difficulty: float, rough_h: float = 0.03) -> Scene:
    scene = Scene()
    amp = rough_h * (0.5 + 0.5 * difficulty)
    cell = 0.2
    nx = max(1, int(round(LENGTH / cell)))
    ny = max(1, int(round(WIDTH / cell)))
    for i in range(nx):
        x0 = i * cell
        x1 = LENGTH if i == nx - 1 else (i + 1) * cell
        for j in range(ny):
            y0 = -WIDTH * 0.5 + j * cell
            y1 = WIDTH * 0.5 if j == ny - 1 else (-WIDTH * 0.5 + (j + 1) * cell)
            top = amp * math.sin(i * 1.7 + j * 2.3) * math.cos(i * 0.9 - j * 1.1)
            bot = -0.02
            hz = max(0.5 * (top - bot), 0.01)
            scene.add(
                f"r_{i}_{j}",
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                0.5 * (top + bot),
                0.5 * (x1 - x0),
                0.5 * (y1 - y0),
                hz,
            )
    scene.floor_z = -0.1
    return scene


def build_flat(_: float) -> Scene:
    scene = Scene()
    _add_ground_slab(scene, "ground", 0.0, LENGTH, -WIDTH * 0.5, WIDTH * 0.5)
    scene.floor_z = -0.1
    return scene


TERRAIN_BUILDERS = [
    ("gap", build_gap),
    ("stones", build_stones),
    ("stones2", build_stones2),
    ("stones1", build_stones1),
    ("bridge", build_bridge),
    ("hurdle", build_hurdle),
    ("ramp", build_ramp),
    ("half_stairs_025", lambda d: build_half_stairs(d, 0.25)),
    ("half_stairs_030", lambda d: build_half_stairs(d, 0.30)),
    ("rough", build_rough),
    ("flat", build_flat),
]


def _row_difficulty(row: int, num_rows: int) -> float:
    """Isaac 用 row/num_rows（最高 0.9）；这里用 row/(num_rows-1) 铺满 0→1，避免末档饱和。"""
    if num_rows <= 1:
        return 1.0
    return row / float(num_rows - 1)


def build_curriculum_scene(builder, num_rows: int) -> Scene:
    """沿 +X 依次延伸 num_rows 段，难度严格递增（易 → 难）。"""
    merged = Scene()
    for row in range(num_rows):
        difficulty = _row_difficulty(row, num_rows)
        seg = builder(difficulty)
        # 段间衔接平台，避免掉进两段之间的缝
        bridge = Scene()
        if row > 0:
            _add_ground_slab(bridge, "link", -0.4, 0.4, -0.6, 0.6)
        merged.extend(bridge.offset_x(row * LENGTH, name_prefix=f"lv{row}_"))
        merged.extend(seg.offset_x(row * LENGTH, name_prefix=f"lv{row}_"))
        # 难度可视化立柱：越高越难
        pillar_h = 0.15 + 0.55 * difficulty
        merged.add(
            f"lv{row}_diff_pillar",
            row * LENGTH + 0.3,
            WIDTH * 0.5 + 0.35,
            0.5 * pillar_h,
            0.08,
            0.08,
            0.5 * pillar_h,
            rgba=f"{0.2 + 0.7 * difficulty:.2f} {0.8 - 0.6 * difficulty:.2f} 0.20 1",
        )
    return merged


def scene_to_xml(terrain_id: int, name: str, scene: Scene, num_rows: int) -> str:
    total_len = num_rows * LENGTH
    center_x = total_len * 0.5
    diffs = ", ".join(f"{_row_difficulty(i, num_rows):.2f}" for i in range(num_rows))
    geoms = [
        f'    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" '
        f'pos="{center_x:.1f} 0 {scene.floor_z:.3f}"/>'
    ]
    for b in scene.boxes:
        rgba = f' rgba="{b.rgba}"' if b.rgba else ""
        geoms.append(
            f'    <geom name="{b.name}" pos="{b.x:.4f} {b.y:.4f} {b.z:.4f}" '
            f'type="box" size="{b.hx:.4f} {b.hy:.4f} {b.hz:.4f}"{rgba}/>'
        )
    body = "\n".join(geoms)
    return f"""<mujoco model="parkour_{terrain_id}_{name}_curriculum{num_rows}">
  <include file="go2.xml"/>

  <statistic center="{center_x:.1f} 0 0.5" extent="{max(8.0, total_len * 0.35):.1f}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1600" offheight="900"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="20 20" reflectance="0.0"/>
  </asset>

  <worldbody>
    <light pos="{center_x:.1f} 0 6" dir="0 0 -1" diffuse="0.9 0.9 0.9" directional="true"/>
    <!-- type={terrain_id} ({name}); curriculum rows={num_rows};
         difficulty = row/(num_rows-1) along +X: [{diffs}]; each segment {LENGTH:.0f}m -->
{body}
  </worldbody>
</mujoco>
"""


def generate_all(out_dir: str, num_rows: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for tid, (name, builder) in enumerate(TERRAIN_BUILDERS):
        scene = build_curriculum_scene(builder, num_rows)
        xml = scene_to_xml(tid, name, scene, num_rows)
        path = os.path.join(out_dir, f"parkour_{tid}_{name}.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)
        written.append(path)
        print(
            f"[ok] {os.path.basename(path)}  "
            f"levels={num_rows}  boxes={len(scene.boxes)}  "
            f"length={num_rows * LENGTH:.0f}m"
        )
    idx_path = os.path.join(out_dir, "parkour_terrains_README.txt")
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write("Parkour MuJoCo scenes — curriculum along +X (easy→hard)\n")
        f.write(f"num_rows={num_rows}  difficulty=row/(num_rows-1)  segment={LENGTH}m\n")
        f.write(
            f"levels: {', '.join(f'{_row_difficulty(i, num_rows):.2f}' for i in range(num_rows))}\n\n"
        )
        for tid, (name, _) in enumerate(TERRAIN_BUILDERS):
            f.write(f"{tid:2d}  parkour_{tid}_{name}.xml\n")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-rows",
        type=int,
        default=10,
        help="curriculum rows (= number of difficulty segments), default 10 like Isaac Gym",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "resources",
            "robots",
            "go2",
            "go2",
        ),
        help="Must be the directory containing go2.xml (meshdir=assets).",
    )
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir)
    n = max(1, int(args.num_rows))
    paths = generate_all(out, n)
    print(f"generated {len(paths)} files -> {out}")


if __name__ == "__main__":
    main()
