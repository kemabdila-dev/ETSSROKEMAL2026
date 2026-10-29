# -*- coding: utf-8 -*-
"""
UTS - 3D Mapping dengan Drone (CoppeliaSim + Python ZMQ Remote API)

Cara pakai (CoppeliaSim harus sudah terbuka):
  python mapping_drone.py build   -> bangun arena + objek di scene (sekali saja, simulasi harus STOP)
  python mapping_drone.py test    -> terbang 1 garis saja, untuk cek orientasi sensor
  python mapping_drone.py full    -> mapping lengkap + post process + plot + evaluasi

Kalau pakai tombol Run di VSCode (tanpa argumen), ubah DEFAULT_MODE di bawah.
"""
import os
import sys
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    try:
        from zmqRemoteApi import RemoteAPIClient
    except ImportError:
        RemoteAPIClient = None

# =====================================================================
#                           KONFIGURASI
# =====================================================================
DEFAULT_MODE = "build"       # "build" | "test" | "full"
DRONE_PATH = "/Quadcopter"   # nama model drone di Scene hierarchy

# ---- Arena ----
ARENA = 5.0                  # sisi arena (m)
WALL_H = 0.5                 # tinggi dinding (m)
WALL_T = 0.1                 # tebal dinding (m)
WALL_COLOR = (0.75, 0.75, 0.95)

# (nama, bentuk, x, y, sx, sy, tinggi, warna RGB). Untuk silinder: sx = diameter.
OBJECTS = [
    ("Box_rendah", "box",       0.5,  0.8, 0.8, 0.8, 0.4, (0.90, 0.50, 0.20)),
    ("Pilar_1",    "box",      -1.2,  0.3, 0.4, 0.4, 1.0, (0.40, 0.50, 0.90)),
    ("Pilar_2",    "box",       1.5, -1.3, 0.4, 0.5, 1.2, (0.30, 0.80, 0.40)),
    ("Box_sedang", "box",      -0.5, -1.5, 0.6, 0.6, 0.7, (0.80, 0.30, 0.50)),
    ("Silinder",   "cylinder",  1.3,  1.5, 0.5, 0.5, 0.8, (0.90, 0.80, 0.20)),
]

# ---- Terbang ----
FLY_Z = 2.0                  # ketinggian terbang (m), harus > objek tertinggi + 0.5
FLY_BOUND = 1.9              # drone terbang di dalam +-FLY_BOUND
LINE_SPACING = 0.7           # jarak antar garis lawnmower (m)
SPEED = 1.0                  # kecepatan target (m/s)
START = (-1.9, -1.9, 0.3)    # posisi awal drone

# ---- Sensor (vision sensor depth, menghadap lurus ke bawah) ----
RES = 128                    # resolusi (RES x RES)
FOV = math.radians(90)       # sudut pandang
NEAR = 0.1
FAR = 5.0
SENSOR_DZ = -0.1             # posisi sensor di bawah badan drone
SCAN_EVERY = 4               # ambil scan tiap N step simulasi
STRIDE = 1                   # 1 = semua pixel, 2 = tiap 2 pixel (lebih ringan)
NOISE_STD = 0.005            # noise depth (m), 0 untuk mematikan
FLIP_X = True                # kalau hasil tercermin, ubah ke False
FLIP_Y = False               # kalau hasil tercermin, ubah ke True

# ---- Post process ----
VOXEL = 0.08                 # ukuran voxel (m)
MIN_PTS_PER_VOXEL = 3        # voxel dengan titik lebih sedikit dibuang (filter noise)
Z_MIN = -0.05
Z_MAX = 1.5                  # buang titik di atas ini (noise / partikel drone)

DRAW_TRAIL_IN_PYTHON = False  # True kalau kamu TIDAK memakai trail.lua
OUT_DIR = "output"

# =====================================================================
#                       VARIABEL GLOBAL
# =====================================================================
client = None
sim = None
drone = None
cam = None
target = None
trail = None
cur = None
dt = 0.05
step_i = 0
path = []
clouds = []
_grid = {}


# =====================================================================
#                       FUNGSI BANTU
# =====================================================================
def connect():
    global client, sim
    if RemoteAPIClient is None:
        raise RuntimeError("Library belum terpasang: pip install coppeliasim-zmqremoteapi-client")
    client = RemoteAPIClient()
    try:
        sim = client.require("sim")
    except AttributeError:
        sim = client.getObject("sim")


def find(paths):
    for p in paths:
        try:
            return sim.getObject(p)
        except Exception:
            pass
    raise RuntimeError("Objek tidak ditemukan. Sudah dicoba: " + ", ".join(paths))


def set_pos(h, ref, p):
    try:
        sim.setObjectPosition(h, ref, p)
    except Exception:
        sim.setObjectPosition(h, p, ref)


def set_ori(h, ref, e):
    try:
        sim.setObjectOrientation(h, ref, e)
    except Exception:
        sim.setObjectOrientation(h, e, ref)


def set_stepping(flag):
    try:
        sim.setStepping(flag)
    except Exception:
        client.setStepping(flag)


# =====================================================================
#                       MODE BUILD (bangun scene)
# =====================================================================
H2 = ARENA / 2.0
WALLS = [
    ("Wall_N", (ARENA, WALL_T, WALL_H), (0.0, H2 - WALL_T / 2, WALL_H / 2)),
    ("Wall_S", (ARENA, WALL_T, WALL_H), (0.0, -(H2 - WALL_T / 2), WALL_H / 2)),
    ("Wall_E", (WALL_T, ARENA - 2 * WALL_T, WALL_H), (H2 - WALL_T / 2, 0.0, WALL_H / 2)),
    ("Wall_W", (WALL_T, ARENA - 2 * WALL_T, WALL_H), (-(H2 - WALL_T / 2), 0.0, WALL_H / 2)),
]


def remove_if_exists(name):
    try:
        h = sim.getObject("/" + name)
    except Exception:
        return
    try:
        sim.removeObjects([h])
    except Exception:
        sim.removeObject(h)


def make_shape(name, kind, size, pos, color):
    ptype = sim.primitiveshape_cuboid if kind == "box" else sim.primitiveshape_cylinder
    h = sim.createPrimitiveShape(ptype, [float(s) for s in size], 0)
    try:
        sim.setObjectAlias(h, name)
    except Exception:
        sim.setObjectName(h, name)
    set_pos(h, -1, [float(p) for p in pos])
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [float(c) for c in color])
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    return h


def build_scene():
    if sim.getSimulationState() != sim.simulation_stopped:
        print("Simulasi masih jalan. Tekan Stop dulu, lalu jalankan mode build lagi.")
        return
    for n in [w[0] for w in WALLS] + [o[0] for o in OBJECTS]:
        remove_if_exists(n)
    for name, size, pos in WALLS:
        make_shape(name, "box", size, pos, WALL_COLOR)
    for (name, kind, x, y, sx, sy, h, col) in OBJECTS:
        size = (sx, sy, h) if kind == "box" else (sx, sx, h)
        make_shape(name, kind, size, (x, y, h / 2.0), col)
    print("Arena + %d objek berhasil dibuat." % len(OBJECTS))
    print("Sekarang SIMPAN scene: File > Save scene as... > drone_mapping.ttt")


# =====================================================================
#                       SENSOR & POINT CLOUD
# =====================================================================
def setup_sensor():
    c = find([DRONE_PATH + "/depthSensor", "/depthSensor",
              DRONE_PATH + "/Vision_sensor", "/Vision_sensor"])
    try:
        sim.setObjectAlias(c, "depthSensor")
    except Exception:
        pass
    sim.setObjectParent(c, drone, False)
    set_pos(c, drone, [0.0, 0.0, SENSOR_DZ])
    set_ori(c, drone, [math.pi, 0.0, 0.0])      # sumbu Z sensor menghadap ke bawah
    sim.setObjectInt32Param(c, sim.visionintparam_resolution_x, RES)
    sim.setObjectInt32Param(c, sim.visionintparam_resolution_y, RES)
    sim.setObjectFloatParam(c, sim.visionfloatparam_near_clipping, NEAR)
    sim.setObjectFloatParam(c, sim.visionfloatparam_far_clipping, FAR)
    sim.setObjectFloatParam(c, sim.visionfloatparam_perspective_angle, FOV)
    try:
        sim.setObjectInt32Param(c, sim.visionintparam_perspective_operation, 1)
    except Exception:
        pass
    return c


SENS = {}


def read_sensor_params():
    """Baca near/far/FOV asli dari sensor, supaya tidak bergantung pada nilai yang kita set."""
    def gf(param, default):
        try:
            r = sim.getObjectFloatParam(cam, param)
            if isinstance(r, (list, tuple)):
                r = r[-1]
            return float(r)
        except Exception:
            return default
    SENS["near"] = gf(sim.visionfloatparam_near_clipping, NEAR)
    SENS["far"] = gf(sim.visionfloatparam_far_clipping, FAR)
    SENS["fov"] = gf(sim.visionfloatparam_perspective_angle, FOV)
    _grid.clear()
    print("Parameter sensor asli: near=%.3f far=%.3f fov=%.1f deg" %
          (SENS["near"], SENS["far"], math.degrees(SENS["fov"])))


def get_grid(w, h):
    if (w, h) not in _grid:
        u = np.arange(w) + 0.5 - w / 2.0
        v = np.arange(h) + 0.5 - h / 2.0
        uu, vv = np.meshgrid(u, v)          # baris 0 = bawah gambar
        f = (w / 2.0) / math.tan(SENS.get("fov", FOV) / 2.0)
        _grid[(w, h)] = (uu / f, vv / f)
    return _grid[(w, h)]


def grab_points():
    """Ambil depth image -> point cloud di koordinat dunia (N x 3)."""
    try:
        sim.handleVisionSensor(cam)
    except Exception:
        pass   # sensor sudah ter-render otomatis tiap step
    buf, res = sim.getVisionSensorDepth(cam)
    w, h = int(res[0]), int(res[1])
    if isinstance(buf, (bytes, bytearray)):
        d = np.frombuffer(bytes(buf), dtype=np.float32)
    else:
        d = np.asarray(buf, dtype=np.float32)
    d = d.reshape(h, w)

    near = SENS.get("near", NEAR)
    far = SENS.get("far", FAR)
    gx, gy = get_grid(w, h)
    z = near + d * (far - near)             # depth metrik (planar)
    valid = d < 0.999                       # buang pixel yang tidak kena apa-apa
    if STRIDE > 1:
        keep = np.zeros_like(valid)
        keep[::STRIDE, ::STRIDE] = True
        valid &= keep
    zz = z[valid].astype(np.float64)
    if NOISE_STD > 0:
        zz = zz + np.random.normal(0.0, NOISE_STD, zz.shape)

    sx = -1.0 if FLIP_X else 1.0
    sy = -1.0 if FLIP_Y else 1.0
    pc = np.stack([sx * gx[valid] * zz, sy * gy[valid] * zz, zz], axis=1)

    M = np.array(sim.getObjectMatrix(cam, -1), dtype=np.float64).reshape(3, 4)
    world = pc @ M[:, :3].T + M[:, 3]
    return world.astype(np.float32)


# =====================================================================
#                       TERBANG
# =====================================================================
def do_step(scan):
    global step_i
    client.step()
    step_i += 1
    p = sim.getObjectPosition(drone, -1)
    path.append(p)
    if trail is not None:
        sim.addDrawingObjectItem(trail, p)
    if scan and step_i % SCAN_EVERY == 0:
        clouds.append(grab_points())
        if len(clouds) == 1:
            print("  scan pertama: %d titik, z dunia min %.2f max %.2f" %
                  (len(clouds[0]), clouds[0][:, 2].min(), clouds[0][:, 2].max()))


def hold(n, scan):
    for _ in range(n):
        do_step(scan)


def fly_to(wp, scan=True):
    global cur
    wp = np.array(wp, dtype=float)
    n = max(int(np.linalg.norm(wp - cur) / (SPEED * dt)), 1)
    for i in range(1, n + 1):
        p = cur + (wp - cur) * i / n
        set_pos(target, -1, p.tolist())
        do_step(scan)
    cur = wp


def lawnmower(bound, spacing, z):
    n_lines = int(math.ceil(2 * bound / spacing)) + 1
    ys = np.linspace(-bound, bound, n_lines)
    wps = []
    for i, y in enumerate(ys):
        xs = (-bound, bound) if i % 2 == 0 else (bound, -bound)
        wps.append((xs[0], float(y), z))
        wps.append((xs[1], float(y), z))
    return wps


# =====================================================================
#                       POST PROCESS
# =====================================================================
def crop_cloud(raw):
    m = ((np.abs(raw[:, 0]) <= ARENA / 2) & (np.abs(raw[:, 1]) <= ARENA / 2) &
         (raw[:, 2] >= Z_MIN) & (raw[:, 2] <= Z_MAX))
    return raw[m]


def voxel_downsample(pts, vs, min_pts=1):
    """Voxel grid: satu titik (centroid) per voxel, voxel jarang dibuang."""
    mn = pts.min(axis=0)
    idx = np.floor((pts - mn) / vs).astype(np.int64)
    dims = idx.max(axis=0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.stack([np.bincount(inv, weights=pts[:, k]) / cnt for k in range(3)], axis=1)
    return out[cnt >= min_pts]


def occupancy_2d(pts, cell=0.05, zmin=0.1, zmax=1.6):
    sl = pts[(pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)]
    edges = np.arange(-ARENA / 2, ARENA / 2 + cell, cell)
    H, _, _ = np.histogram2d(sl[:, 0], sl[:, 1], bins=[edges, edges])
    return (H > 0).T


def evaluate(pts):
    lines = ["Evaluasi tinggi objek (titik 99-persentil z di dalam footprint objek)",
             "%-12s %10s %12s %10s" % ("Objek", "Asli (m)", "Terukur (m)", "Error (cm)")]
    for (name, kind, x, y, sx, sy, h, col) in OBJECTS:
        m = (np.abs(pts[:, 0] - x) <= sx / 2) & (np.abs(pts[:, 1] - y) <= sy / 2)
        if m.sum() < 10:
            lines.append("%-12s tidak terdeteksi" % name)
            continue
        zt = float(np.percentile(pts[m, 2], 99))
        lines.append("%-12s %10.2f %12.3f %10.1f" % (name, h, zt, (zt - h) * 100))
    return "\n".join(lines)


def draw_gt(ax):
    for (name, kind, x, y, sx, sy, h, col) in OBJECTS:
        ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, fill=False,
                               ec="red", ls="--", lw=1.0))


def make_plots(raw, vox, pth, show=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    pth = np.array(pth)
    half = ARENA / 2

    # 1) Jalur drone
    fig1 = plt.figure(figsize=(11, 5))
    ax = fig1.add_subplot(1, 2, 1, projection="3d")
    ax.plot(pth[:, 0], pth[:, 1], pth[:, 2], "b", lw=1.5, label="Jalur drone")
    ax.scatter(pth[0, 0], pth[0, 1], pth[0, 2], c="g", s=50, label="Start")
    ax.scatter(pth[-1, 0], pth[-1, 1], pth[-1, 2], c="r", s=50, label="Finish")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Jalur drone (3D)"); ax.legend()
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.plot(pth[:, 0], pth[:, 1], "b", lw=1)
    ax2.set_aspect("equal"); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_title("Jalur drone (tampak atas)")
    fig1.tight_layout(); fig1.savefig(os.path.join(OUT_DIR, "1_jalur_drone.png"), dpi=150)

    # 2) Scan 2D (occupancy grid) + ground truth
    fig2, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(occupancy_2d(raw), origin="lower", extent=[-half, half, -half, half], cmap="gray_r")
    ax.plot(pth[:, 0], pth[:, 1], "b", lw=0.6, alpha=0.6, label="Jalur drone")
    draw_gt(ax)
    ax.plot([], [], "r--", label="Ground truth objek")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(loc="upper right")
    ax.set_title("Peta 2D (occupancy grid, irisan z 0.1-1.6 m)")
    fig2.tight_layout(); fig2.savefig(os.path.join(OUT_DIR, "2_scan_2d.png"), dpi=150)

    # 3) Scan 3D mentah
    sub = raw[np.random.default_rng(0).choice(len(raw), min(len(raw), 60000), replace=False)]
    fig3 = plt.figure(figsize=(7, 6))
    ax = fig3.add_subplot(projection="3d")
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c="gray", s=1)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(0, 2.0)
    ax.set_box_aspect((1, 1, 0.5)); ax.view_init(30, -60)
    ax.set_title("Scan 3D (point cloud mentah, %d titik)" % len(raw))
    fig3.tight_layout(); fig3.savefig(os.path.join(OUT_DIR, "3_scan_3d_mentah.png"), dpi=150)

    # 4) Voxel + color mapping ketinggian + jalur
    fig4 = plt.figure(figsize=(8, 7))
    ax = fig4.add_subplot(projection="3d")
    sc = ax.scatter(vox[:, 0], vox[:, 1], vox[:, 2], c=vox[:, 2], cmap="viridis", s=8)
    ax.plot(pth[:, 0], pth[:, 1], pth[:, 2], "b", lw=1.5, label="Jalur drone")
    fig4.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Tinggi (m)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(0, 2.0)
    ax.set_box_aspect((1, 1, 0.5)); ax.view_init(30, -60); ax.legend()
    ax.set_title("3D Map - voxel %.0f cm, color map ketinggian (%d voxel)" % (VOXEL * 100, len(vox)))
    fig4.tight_layout(); fig4.savefig(os.path.join(OUT_DIR, "4_voxel_color_map.png"), dpi=150)

    if show:
        plt.show()


def postprocess_and_report(mode):
    raw_all = np.vstack(clouds)
    raw = crop_cloud(raw_all)
    vox = voxel_downsample(raw, VOXEL, MIN_PTS_PER_VOXEL)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "cloud_raw.npy"), raw)
    np.save(os.path.join(OUT_DIR, "cloud_voxel.npy"), vox)
    np.save(os.path.join(OUT_DIR, "path.npy"), np.array(path))

    ringkas = [
        "Mode               : %s" % mode,
        "Jumlah scan        : %d" % len(clouds),
        "Titik mentah       : %d" % len(raw_all),
        "Setelah crop       : %d" % len(raw),
        "Setelah voxel %.0f cm: %d (%.1fx lebih sedikit)" % (VOXEL * 100, len(vox), len(raw) / max(len(vox), 1)),
        "Panjang jalur (m)  : %.1f" % float(np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))),
        "",
        evaluate(raw),
    ]
    txt = "\n".join(ringkas)
    print("\n" + txt)
    with open(os.path.join(OUT_DIR, "hasil.txt"), "w") as f:
        f.write(txt)
    make_plots(raw, vox, path, show=True)


# =====================================================================
#                       MODE TEST / FULL
# =====================================================================
def run_mapping(mode):
    global drone, cam, target, cur, dt, trail, step_i, path, clouds
    step_i = 0
    path = []
    clouds = []

    drone = find([DRONE_PATH])
    cam = setup_sensor()
    set_pos(drone, -1, list(START))

    set_stepping(True)
    sim.startSimulation()
    try:
        client.step() 
        read_sensor_params()      # supaya script drone sempat init (melepas target)
        target = find(["/target", DRONE_PATH + "/target", "/Quadcopter_target", "/Quadricopter_target"])
        dt = sim.getSimulationTimeStep()
        cur = np.array(sim.getObjectPosition(target, -1), dtype=float)
        if DRAW_TRAIL_IN_PYTHON:
            trail = sim.addDrawingObject(sim.drawing_linestrip, 4, 0, -1, 100000, [0, 0, 1])

        if mode == "test":
            wps = [(-FLY_BOUND, 0.0, FLY_Z), (FLY_BOUND, 0.0, FLY_Z)]
        else:
            wps = lawnmower(FLY_BOUND, LINE_SPACING, FLY_Z)
        print("Terbang: %d waypoint, dt=%.3f" % (len(wps), dt))

        fly_to((wps[0][0], wps[0][1], FLY_Z), scan=False)   # naik ke ketinggian terbang
        hold(40, False)                                      # tunggu drone stabil
        for i, wp in enumerate(wps):
            fly_to(wp, scan=True)
            print("  waypoint %d/%d selesai" % (i + 1, len(wps)))
        hold(20, True)

        postprocess_and_report(mode)
    finally:
        sim.stopSimulation()
        time.sleep(1.0)
        try:
            set_stepping(False)
        except Exception:
            pass


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else DEFAULT_MODE
    connect()
    if mode == "build":
        build_scene()
    elif mode in ("test", "full"):
        run_mapping(mode)
    else:
        print("Mode tidak dikenal. Pakai: build | test | full")


if __name__ == "__main__":
    main()
