"""BEV 위에서 자차 차로(2차로)의 중심선을 추정한다.

트랙은 2차로이고 자차는 왼쪽에서 두 번째 차로를 달린다.
따라서 자차 차로의 경계는

    왼쪽 = center_line (점선 중앙선)
    오른쪽 = 바깥쪽 실선

이고, 모델이 내주는 클래스는 이렇게 대응된다.

    center_line : 중앙선 점선 조각들 (조각 하나가 detection 하나)
    lane1       : 1차로(왼쪽 차로) 영역
    lane2       : 2차로(자차 차로) 영역

기존 방식은 lane2 영역 외곽선의 좌우 끝 중점을 차로 중심으로 썼는데,
교차 구간에서 lane2 마스크가 주차장 쪽으로 새면 왼쪽 끝이 x=0 까지 끌려가
중심이 통째로 왼쪽으로 밀린다. 좌측 쏠림과 이탈의 원인이다.

여기서는 왼쪽 경계와 오른쪽 경계를 따로 추정하고, 둘의 간격이 차로 폭으로
말이 될 때만 중점을 쓴다. 한쪽만 믿을 만하면 차로 폭의 절반을 더하거나 뺀다.
center_line 은 점선이라 특정 행에 조각이 없을 수 있으므로, 조각들의 점을 모두
모아 x = f(y) 곡선으로 맞춘 뒤 원하는 행에서 값을 읽는다.
"""

import math

import cv2
import numpy as np

# BEV 좌표계: 640x480, y=0 이 가장 먼 곳, y=479 가 차량 앞범퍼 부근.
BEV_W = 640
BEV_H = 480

# 차로 폭 [BEV 픽셀].
# 트랙 텍스처의 차선 위치를 직접 재보니 차로 폭이 3.03~3.35m 였다.
# 가로 축척 103.6 px/m 을 곱하면 314~347px 이므로 그 중간을 쓴다.
LANE_WIDTH = 328.0

# 좌우 경계를 동시에 얻었을 때, 간격이 이 범위를 벗어나면 한쪽이 오검출이다.
LANE_WIDTH_MIN = 220.0
LANE_WIDTH_MAX = 400.0

# 경계 추정에 쓰는 세로 밴드 두께 [px]
BAND_HALF = 6

# 한 행에서 영역 마스크를 덩어리로 나눌 때, 이만큼 떨어지면 다른 덩어리로 본다 [px]
RUN_GAP = 15
RUN_MIN_WIDTH = 40                # 이보다 좁으면 노이즈
RUN_MAX_WIDTH_RATIO = 1.15        # 차로 폭 상한의 이 배를 넘으면 마스크가 샌 것
MAX_TILT = math.radians(65)       # 기울기 보정 상한. 이보다 누우면 더 늘리지 않는다

# center_line 곡선 맞춤 조건
FIT_MIN_POINTS = 8      # 이보다 적으면 곡선을 믿지 않는다
FIT_MIN_SPREAD = 60     # y 방향으로 이만큼은 퍼져 있어야 기울기가 의미 있다
FIT_QUAD_SPREAD = 150   # 이만큼 퍼지면 2차식까지 쓴다. 코너에서 곡률을 살리려면 낮아야 한다
FIT_MAX_RESIDUAL = 45   # 잔차가 이보다 크면 맞춤 실패로 본다 [px]

# 2차로 마스크의 왼쪽 끝이 중앙선과 이만큼 안에서 일치할 때만 믿는다 [px]
FIT_AGREE_TOL = 90.0

# 곡선 맞춤 없이 행별로 중앙선 위치를 뽑을 때의 조건.
# 점선이라 해당 행에 조각이 없을 수 있으므로 위아래로 넉넉히 본다.
CL_ROW_WINDOW = 70      # 이 행 범위 안의 중앙선 점을 모은다 [px]
CL_MIN_POINTS = 4       # 이보다 적으면 쓰지 않는다
CL_GROUP_GAP = 60       # 이만큼 떨어지면 다른 선으로 본다 [px]


def perspective_matrix(src_mat):
    """원본 이미지 -> BEV 호모그래피. lane_info_extractor 의 dst_mat 과 같은 규약."""
    dst_mat = [[round(BEV_W * 0.3), 0], [round(BEV_W * 0.7), 0],
               [round(BEV_W * 0.7), BEV_H], [round(BEV_W * 0.3), BEV_H]]
    return cv2.getPerspectiveTransform(np.float32(src_mat), np.float32(dst_mat))


def points_to_bev(points, matrix):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def mask_to_bev(polygons, matrix):
    """폴리곤들을 채운 마스크를 BEV 로 옮긴다. 영역 클래스(lane1/lane2)용."""
    img = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(img, [pts], 255)
    return cv2.warpPerspective(img, matrix, (BEV_W, BEV_H), flags=cv2.INTER_NEAREST)


def fit_center_line(bev_points):
    """중앙선 점들을 x = f(y) 로 맞춘다. 실패하면 None."""
    if len(bev_points) < FIT_MIN_POINTS:
        return None

    pts = bev_points[(bev_points[:, 0] >= 0) & (bev_points[:, 0] < BEV_W) &
                     (bev_points[:, 1] >= 0) & (bev_points[:, 1] < BEV_H)]
    if len(pts) < FIT_MIN_POINTS:
        return None

    ys, xs = pts[:, 1], pts[:, 0]
    if ys.max() - ys.min() < FIT_MIN_SPREAD:
        return None

    degree = 2 if (ys.max() - ys.min()) >= FIT_QUAD_SPREAD else 1

    # 한 번 맞춘 뒤 잔차가 큰 점을 버리고 다시 맞춘다.
    # 반대편 차선이나 주차장 표시가 center_line 으로 섞여 들어오는 경우를 걸러낸다.
    coeff = np.polyfit(ys, xs, degree)
    residual = np.abs(np.polyval(coeff, ys) - xs)
    keep = residual <= max(FIT_MAX_RESIDUAL, np.median(residual) * 2.5)
    if keep.sum() >= FIT_MIN_POINTS and keep.sum() < len(pts):
        ys, xs = ys[keep], xs[keep]
        if ys.max() - ys.min() < FIT_MIN_SPREAD:
            return None
        degree = 2 if (ys.max() - ys.min()) >= FIT_QUAD_SPREAD else 1
        coeff = np.polyfit(ys, xs, degree)
        residual = np.abs(np.polyval(coeff, ys) - xs)

    if np.sqrt(np.mean(residual ** 2)) > FIT_MAX_RESIDUAL:
        return None
    return coeff


def region_edges(bev_mask, row, ref_x=None, tilt=0.0):
    """영역 마스크에서 해당 행의 좌우 끝. 픽셀이 없으면 (None, None).

    행 전체의 최소/최대를 그냥 쓰면, 교차 구간에서 마스크가 주차장이나 옆 도로로
    새는 순간 끝점이 이미지 가장자리까지 끌려가 차로 중심이 통째로 밀린다.
    그래서 끊긴 덩어리로 나눈 뒤 차로 하나로 볼 만한 폭을 가진 덩어리만 쓰고,
    그 중 ref_x(직전 행에서 찾은 중심)에 가장 가까운 것을 고른다.

    tilt 는 그 행에서 차로가 기울어진 각도 [rad] 다. BEV 에서 차로가 기울면
    가로로 자른 폭이 실제 폭의 1/cos(tilt) 배가 된다. 60도면 두 배다.
    이걸 반영하지 않으면 급코너에서 멀쩡한 행을 '너무 넓다'며 통째로 버리고,
    그 결과 조향을 놓아버린다.
    """
    band = bev_mask[max(0, row - BAND_HALF): row + BAND_HALF, :]
    if band.size == 0:
        return None, None
    cols = np.where(band.max(axis=0) > 0)[0]
    if len(cols) == 0:
        return None, None

    if ref_x is None:
        ref_x = BEV_W / 2.0

    # 기울어진 만큼 폭 상한을 늘린다
    widen = 1.0 / max(math.cos(min(abs(tilt), MAX_TILT)), 0.35)

    best = None
    for run in np.split(cols, np.where(np.diff(cols) > RUN_GAP)[0] + 1):
        width = float(run[-1] - run[0])
        if width < RUN_MIN_WIDTH or width > LANE_WIDTH_MAX * RUN_MAX_WIDTH_RATIO * widen:
            continue
        if run[0] <= ref_x <= run[-1]:
            dist = 0.0
        else:
            dist = min(abs(run[0] - ref_x), abs(run[-1] - ref_x))
        if best is None or dist < best[0]:
            best = (dist, float(run[0]), float(run[-1]))

    if best is None:
        return None, None
    return best[1], best[2]


class LaneCenterEstimator:
    """한 프레임의 detection 에서 목표 행들의 차로 중심 x 를 만든다."""

    def __init__(self, src_mat, lane_width=LANE_WIDTH, center_offset=0.0):
        self.matrix = perspective_matrix(src_mat)
        self.lane_width = float(lane_width)
        # 양수면 차로 중심보다 오른쪽을 노린다 [BEV 픽셀]. 미세 조정용.
        self.center_offset = float(center_offset)
        # 디버그 이미지를 다시 계산하지 않도록 마지막 프레임의 BEV 레이어를 남겨둔다
        self._last = None
        # 차로가 BEV 에서 기울어진 각도 [rad]. 폭 판정을 보정하는 데 쓴다.
        self.tilt = 0.0

    def estimate(self, detections, rows, ref_x0=None):
        """(centers, debug) 를 돌려준다.

        centers 는 {row: x} 이고, 추정에 실패한 행은 아예 넣지 않는다.
        debug 는 어느 근거를 썼는지 세어둔 것.
        ref_x0 는 직전 프레임에서 추적하던 차로의 위치다. 이걸 주면 같은 차로를
        계속 따라간다. 주지 않으면 차량 중심에서 시작한다.
        """
        center_pts, lane1_polys, lane2_polys = [], [], []
        for det in detections:
            mask = getattr(det, 'mask', None)
            if mask is None or not mask.data:
                continue
            poly = [(p.x, p.y) for p in mask.data]
            if det.class_name == 'center_line':
                center_pts.append(poly)
            elif det.class_name == 'lane1':
                lane1_polys.append(poly)
            elif det.class_name == 'lane2':
                lane2_polys.append(poly)

        fit, cl_bev = None, None
        if center_pts:
            cl_bev = points_to_bev(np.vstack([np.asarray(p) for p in center_pts]), self.matrix)
            cl_bev = cl_bev[(cl_bev[:, 0] >= 0) & (cl_bev[:, 0] < BEV_W) &
                            (cl_bev[:, 1] >= 0) & (cl_bev[:, 1] < BEV_H)]
            fit = fit_center_line(cl_bev)

        lane1_bev = mask_to_bev(lane1_polys, self.matrix) if lane1_polys else None
        lane2_bev = mask_to_bev(lane2_polys, self.matrix) if lane2_polys else None
        # 라벨과 무관하게 '차로 영역'을 합쳐 둔다.
        # 어떤 구간에서는 모델이 자차 차로를 lane1 으로 붙이기도 하는데,
        # 자차 차로는 정의상 차가 들어 있는 차로이므로 라벨보다 위치가 확실하다.
        if lane1_bev is not None and lane2_bev is not None:
            lanes_bev = cv2.bitwise_or(lane1_bev, lane2_bev)
        else:
            lanes_bev = lane1_bev if lane1_bev is not None else lane2_bev

        # 가까운 행부터 훑으면서, 직전 행에서 찾은 중심을 다음 행의 기준으로 넘긴다.
        # 자차가 있는 차로에서 출발해 앞으로 따라가는 셈이라, 옆 차로나 갈라지는
        # 도로의 마스크를 잘못 물고 가는 일이 줄어든다.
        # 중앙선 곡선이 있으면 그 기울기로, 없으면 직전 값을 유지한다
        if fit is not None:
            slope = float(np.polyval(np.polyder(fit, 1), BEV_H * 0.7))
            self.tilt = math.atan(slope)

        centers, sources = {}, {}
        # 직전 프레임에서 보던 차로에서 이어서 찾는다. 매번 차량 중심에서 시작하면
        # 두 차로가 모두 후보일 때 프레임마다 다른 차로를 물어 값이 통째로 튄다.
        ref_x = BEV_W / 2.0 if ref_x0 is None else float(ref_x0)
        for row in sorted(rows, reverse=True):
            left = self.left_boundary(row, fit, lane1_bev, lane2_bev, ref_x, cl_bev)
            right = self.right_boundary(row, lane2_bev, ref_x)
            x, src = self.combine(left, right,
                                  cl_bev is not None and len(cl_bev) >= CL_MIN_POINTS)
            if x is None:
                # 마지막 수단: 차가 들어 있는 차로 영역의 좌우 끝을 그대로 쓴다.
                x, src = self.ego_lane_center(lanes_bev, row, ref_x)
            if x is None:
                continue
            x += self.center_offset
            if 0.0 <= x < BEV_W:
                centers[row] = x
                sources[row] = src
                ref_x = x

        self._last = (fit, lane1_bev, lane2_bev)
        return centers, {'fit': fit is not None, 'sources': sources,
                         'n_center_line': len(center_pts)}

    def debug_image(self, detections=None):
        """마지막 estimate 결과를 mono8 한 장으로 그린다.

        2차로 영역은 어둡게, 1차로 영역은 더 어둡게, 중앙선 곡선은 밝게.
        """
        img = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
        if self._last is None:
            return img
        fit, lane1_bev, lane2_bev = self._last
        if lane1_bev is not None:
            img[lane1_bev > 0] = 60
        if lane2_bev is not None:
            img[lane2_bev > 0] = 110
        if fit is not None:
            rows = np.arange(BEV_H)
            cols = np.polyval(fit, rows)
            valid = (cols >= 0) & (cols < BEV_W)
            img[rows[valid], cols[valid].astype(np.int32)] = 255
        return img

    @staticmethod
    def center_line_x(cl_bev, row, ref_x=None):
        """해당 행 부근 중앙선 점들의 x 중앙값. 곡선 맞춤이 실패해도 쓸 수 있다.

        점선이라 특정 행에 조각이 없을 수 있어 위아래로 넉넉히 본다.
        조각이 여러 개 흩어져 있으면 ref_x 에 가까운 쪽을 고른다.
        """
        if cl_bev is None or len(cl_bev) == 0:
            return None
        sel = cl_bev[np.abs(cl_bev[:, 1] - row) <= CL_ROW_WINDOW]
        if len(sel) < CL_MIN_POINTS:
            return None
        xs = np.sort(sel[:, 0])
        groups = np.split(xs, np.where(np.diff(xs) > CL_GROUP_GAP)[0] + 1)
        groups = [g for g in groups if len(g) >= CL_MIN_POINTS]
        if not groups:
            return None
        if ref_x is None:
            ref_x = BEV_W / 2.0
        best = min(groups, key=lambda g: abs(float(np.median(g)) - ref_x))
        return float(np.median(best))

    def left_boundary(self, row, fit, lane1_bev, lane2_bev, ref_x=None, cl_bev=None):
        """자차 차로의 왼쪽 경계(=중앙선) x.

        2차로 영역의 왼쪽 끝을 우선한다. 행마다 따로 재기 때문에 곡선을 그대로
        담아내는 반면, 중앙선 곡선맞춤은 점선 조각이 세로로 짧게 퍼진 코너에서
        1차식으로 떨어져 곡률이 통째로 사라진다. 그러면 제어기가 코너를 직선으로
        보고 덜 꺾는다.
        다만 2차로 마스크는 교차 구간에서 주차장 쪽으로 새므로, 중앙선 곡선이
        있으면 그것과 크게 어긋나지 않을 때만 채택한다.
        """
        fit_x = None
        if fit is not None:
            x = float(np.polyval(fit, row))
            if 0.0 <= x < BEV_W:
                fit_x = x

        # 곡선 맞춤이 실패해도 쓸 수 있는, 행별 중앙선 위치
        cl_x = self.center_line_x(cl_bev, row, ref_x if ref_x is None else ref_x - self.lane_width / 2.0)
        anchor = fit_x if fit_x is not None else cl_x

        if lane2_bev is not None:
            left, _ = region_edges(lane2_bev, row, ref_x, self.tilt)
            if left is not None and left > 2:
                if anchor is None or abs(left - anchor) <= FIT_AGREE_TOL:
                    return left, 'lane2L'

        if fit_x is not None:
            return fit_x, 'fit'
        if cl_x is not None:
            return cl_x, 'cl'

        # 1차로 영역의 오른쪽 끝도 중앙선이다. 다만 자차가 차로 안에 있다는 보장이
        # 없으므로, 중앙선 점이 실제로 그 근처에 있을 때만 믿는다.
        if lane1_bev is not None:
            _, right = region_edges(lane1_bev, row, ref_x, self.tilt)
            if right is not None and right < BEV_W - 2:
                return right, 'lane1'
        return None, None

    def ego_lane_center(self, lanes_bev, row, ref_x):
        """차가 들어 있는 차로 영역의 중심. 클래스 라벨을 믿을 수 없을 때 쓴다.

        lane1/lane2 를 합친 마스크에서 ref_x 를 품고 있는 덩어리를 고르고,
        그 폭이 차로 하나로 말이 될 때만 중심을 돌려준다.
        """
        if lanes_bev is None:
            return None, None
        left, right = region_edges(lanes_bev, row, ref_x, self.tilt)
        if left is None:
            return None, None
        width = (right - left) * math.cos(min(abs(self.tilt), MAX_TILT))
        if not (LANE_WIDTH_MIN <= width <= LANE_WIDTH_MAX):
            return None, None
        if not (left - 20 <= ref_x <= right + 20):
            # 자차가 그 덩어리 안에 있지 않으면 자차 차로가 아니다
            return None, None
        return (left + right) / 2.0, 'ego'

    def right_boundary(self, row, lane2_bev, ref_x=None):
        """자차 차로의 오른쪽 경계(=바깥쪽 실선) x."""
        if lane2_bev is None:
            return None, None
        _, right = region_edges(lane2_bev, row, ref_x, self.tilt)
        if right is None or right > BEV_W - 3:
            # 이미지 끝에 붙었으면 실제 경계가 아니라 잘린 것이다.
            return None, None
        return right, 'lane2R'

    def combine(self, left, right, has_center_line=False):
        """좌우 경계에서 차로 중심을 만든다. 근거가 약하면 값을 내지 않는다.

        추정을 못 하는 것보다 나쁜 것은 틀린 값을 자신 있게 내는 것이다.
        차가 도로를 벗어나면 lane2(자차 차로)가 통째로 사라지는데, 이때 lane1 의
        오른쪽 끝만 보고 차로를 지어내면 "거의 중앙"이라는 오답이 나와 그대로
        잔디 위를 계속 달리게 된다. 그래서 lane1 단독은 근거로 쓰지 않는다.
        """
        left_x, left_src = left
        right_x, right_src = right

        if right_x is not None:
            if left_x is not None:
                width = (right_x - left_x) * math.cos(min(abs(self.tilt), MAX_TILT))
                if LANE_WIDTH_MIN <= width <= LANE_WIDTH_MAX:
                    # 좌우가 모두 잡히고 폭도 말이 되면 중점이 가장 정확하다.
                    # 차로 폭 상수가 조금 틀려도 영향을 받지 않는다.
                    return (left_x + right_x) / 2.0, f'{left_src}+{right_src}'
            # 바깥쪽 실선은 잔디와 맞닿은 연속선이라 거의 놓치지 않는다.
            # 점선 중앙선보다 훨씬 안정적이므로, 하나만 믿어야 한다면 이쪽이다.
            return right_x - self.lane_width / 2.0, right_src

        # 바깥쪽을 못 봤을 때는 중앙선에 기댄다. 중앙선은 그 자체로 자차 차로의
        # 왼쪽 경계이므로 단독으로도 쓸 수 있다.
        if left_x is not None and left_src in ('fit', 'cl'):
            return left_x + self.lane_width / 2.0, left_src
        # lane1 오른쪽 끝은 중앙선을 실제로 본 프레임에서만 인정한다.
        # 도로를 벗어났을 때 lane1 만 보고 차로를 지어내면 그럴듯한 오답이 나온다.
        if left_x is not None and left_src == 'lane1' and has_center_line:
            return left_x + self.lane_width / 2.0, left_src
        return None, None
