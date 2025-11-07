mport numpy as np
import cv2

# 画布
CANVAS_WIDTH, CANVAS_HEIGHT = 1280, 720
MARGIN = 20  # 画布四周留白

# 单应性矩阵 H（你的数值）
H = np.array([
    [-944.336237,   -87.4102614,   -4.95230579],
    [ -62.4513781,  874.191558,   134.016380  ],
    [  -0.00962922264, -0.142556057, 1.0]
], dtype=float)

# 五个矩形：世界坐标 (x_min, y_min, x_max, y_max) 单位：m
rects = {
    "PCB1": (0.6016, -0.1700, 0.8266, -0.1100),
    "PCB2": (0.2961, -0.1700, 0.5823, -0.1100),
    "PCB3": (0.0948, -0.4301, 0.2552, -0.3701),
    "Battery": (0.5758, -0.0492, 0.7318, 0.1332),
    "M": (0.2341, -0.0561, 0.5351, 0.1451),
}

def rect_corners(xmin, ymin, xmax, ymax):
    # 顺时针：左下、左上、右上、右下（和你之前保持一致）
    return np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]
    ], dtype=np.float32)

# 1) 组装所有世界坐标点
world_polys = {}
all_world_pts = []
for name, (xmin, ymin, xmax, ymax) in rects.items():
    corners = rect_corners(xmin, ymin, xmax, ymax)
    world_polys[name] = corners
    all_world_pts.append(corners)
all_world_pts = np.concatenate(all_world_pts, axis=0)  # (N,2)

# 2) 透视投影到像素（未缩放/平移前，可能为负或超界）
#    OpenCV 接口：输入 (N,1,2) float32，输出 (N,1,2)
pts_in = all_world_pts.reshape(-1,1,2).astype(np.float32)
pts_proj = cv2.perspectiveTransform(pts_in, H).reshape(-1,2)  # (N,2)

# 3) 计算统一的缩放和平移，使所有点都落入画布（留白 MARGIN）
u_min, v_min = pts_proj.min(axis=0)
u_max, v_max = pts_proj.max(axis=0)
range_u = u_max - u_min
range_v = v_max - v_min

# 等比缩放，优先保证都能放进画布
scale_x = (CANVAS_WIDTH  - 2*MARGIN) / range_u
scale_y = (CANVAS_HEIGHT - 2*MARGIN) / range_v
scale = min(scale_x, scale_y)

# 平移量（把最小值对齐到 MARGIN 处）
tx = MARGIN - scale * u_min
ty = MARGIN - scale * v_min

def apply_post_transform(uv):
    """对投影后的 (u,v) 做统一缩放+平移"""
    u = scale * uv[:,0] + tx
    v = scale * uv[:,1] + ty
    return np.stack([u, v], axis=1)

# 4) 逐个多边形投影+归一化并绘制
canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255

offset = 0
for name, world_corners in world_polys.items():
    # 该多边形对应的投影（从 pts_proj 中切片取回）
    poly_proj = pts_proj[offset:offset+4]  # (4,2)
    offset += 4

    # 归一化（缩放+平移）
    poly_img = apply_post_transform(poly_proj).round().astype(np.int32).reshape(-1,1,2)

    # 画透视四边形
    cv2.polylines(canvas, [poly_img], isClosed=True, color=(0,0,255), thickness=2)

    # 画轴对齐外接矩形
    x, y, w, h = cv2.boundingRect(poly_img)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0,180,0), 1)

    # 标注
    cv2.putText(canvas, name, (x+3, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2, cv2.LINE_AA)

# 可视化
cv2.imshow("Projected & Fitted Rectangles", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果需要后续直接用“只需两个点”的外接框像素坐标，可在循环中保存 (x, y, x+w, y+h)。
