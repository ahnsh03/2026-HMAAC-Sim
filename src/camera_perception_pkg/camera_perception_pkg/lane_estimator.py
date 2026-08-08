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

# center_line 곡선 맞춤 조건
FIT_MIN_POINTS = 8      # 이보다 적으면 곡선을 믿지 않는다
FIT_MIN_SPREAD = 60     # y 방향으로 이만큼은 퍼져 있어야 기울기가 의미 있다
FIT_QUAD_SPREAD = 150   # 이만큼 퍼지면 2차식까지 쓴다. 코너에서 곡률을 살리려면 낮아야 한다
FIT_MAX_RESIDUAL = 45   # 잔차가 이보다 크면 맞춤 실패로 본다 [px]

# 2차로 마스크의 왼쪽 끝이 중앙선 곡선과 이만큼 안에서 일치할 때만 믿는다 [px]
FIT_AGREE_TOL = 90.0


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


def region_edges(bev_mask, row, ref_x=None):
    """영역 마스크에서 해당 행의 좌우 끝. 픽셀이 없으면 (None, None).

    행 전체의 최소/최대를 그냥 쓰면, 교차 구간에서 마스크가 주차장이나 옆 도로로
    새는 순간 끝점이 이미지 가장자리까지 끌려가 차로 중심이 통째로 밀린다.
    그래서 끊긴 덩어리로 나눈 뒤 차로 하나로 볼 만한 폭을 가진 덩어리만 쓰고,
    그 중 ref_x(직전 행에서 찾은 중심)에 가장 가까운 것을 고른다.
    """
    band = bev_mask[max(0, row - BAND_HALF): row + BAND_HALF, :]
    if band.size == 0:
        return None, None
    cols = np.where(band.max(axis=0) > 0)[0]
    if len(cols) == 0:
        return None, None

    if ref_x is None:
        ref_x = BEV_W / 2.0

    best = None
    for run in np.split(cols, np.where(np.diff(cols) > RUN_GAP)[0] + 1):
        width = float(run[-1] - run[0])
        if width < RUN_MIN_WIDTH or width > LANE_WIDTH_MAX * RUN_MAX_WIDTH_RATIO:
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

    def estimate(self, detections, rows):
        """(centers, debug) 를 돌려준다.

        centers 는 {row: x} 이고, 추정에 실패한 행은 아예 넣지 않는다.
        debug 는 어느 근거를 썼는지 세어둔 것.
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

        fit = None
        if center_pts:
            bev_pts = points_to_bev(np.vstack([np.asarray(p) for p in center_pts]), self.matrix)
            fit = fit_center_line(bev_pts)

        lane1_bev = mask_to_bev(lane1_polys, self.matrix) if lane1_polys else None
        lane2_bev = mask_to_bev(lane2_polys, self.matrix) if lane2_polys else None

        # 가까운 행부터 훑으면서, 직전 행에서 찾은 중심을 다음 행의 기준으로 넘긴다.
        # 자차가 있는 차로에서 출발해 앞으로 따라가는 셈이라, 옆 차로나 갈라지는
        # 도로의 마스크를 잘못 물고 가는 일이 줄어든다.
        centers, sources = {}, []
        ref_x = BEV_W / 2.0
        for row in sorted(rows, reverse=True):
            left = self.left_boundary(row, fit, lane1_bev, lane2_bev, ref_x)
            right = self.right_boundary(row, lane2_bev, ref_x)
            x, src = self.combine(left, right)
            if x is None:
                continue
            x += self.center_offset
            if 0.0 <= x < BEV_W:
                centers[row] = x
                sources.append(src)
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

    def left_boundary(self, row, fit, lane1_bev, lane2_bev, ref_x=None):
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

        if lane2_bev is not None:
            left, _ = region_edges(lane2_bev, row, ref_x)
            if left is not None and left > 2:
                if fit_x is None or abs(left - fit_x) <= FIT_AGREE_TOL:
                    return left, 'lane2L'

        if fit_x is not None:
            return fit_x, 'fit'

        # 1차로 영역의 오른쪽 끝도 중앙선이지만, 자차가 차로 안에 있다는 보장이 없어
        # 단독으로는 쓰지 않는다 (combine 에서 lane2R 과 함께일 때만 채택).
        if lane1_bev is not None:
            _, right = region_edges(lane1_bev, row, ref_x)
            if right is not None and right < BEV_W - 2:
                return right, 'lane1'
        return None, None

    def right_boundary(self, row, lane2_bev, ref_x=None):
        """자차 차로의 오른쪽 경계(=바깥쪽 실선) x."""
        if lane2_bev is None:
            return None, None
        _, right = region_edges(lane2_bev, row, ref_x)
        if right is None or right > BEV_W - 3:
            # 이미지 끝에 붙었으면 실제 경계가 아니라 잘린 것이다.
            return None, None
        return right, 'lane2R'

    def combine(self, left, right):
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
                width = right_x - left_x
                if LANE_WIDTH_MIN <= width <= LANE_WIDTH_MAX:
                    # 좌우가 모두 잡히고 폭도 말이 되면 중점이 가장 정확하다.
                    # 차로 폭 상수가 조금 틀려도 영향을 받지 않는다.
                    return (left_x + right_x) / 2.0, f'{left_src}+{right_src}'
            # 바깥쪽 실선은 잔디와 맞닿은 연속선이라 거의 놓치지 않는다.
            # 점선 중앙선보다 훨씬 안정적이므로, 하나만 믿어야 한다면 이쪽이다.
            return right_x - self.lane_width / 2.0, right_src

        # 바깥쪽을 못 봤을 때만 중앙선에 기댄다.
        # lane2 영역의 왼쪽 끝(lane2L)은 교차 구간에서 잘 새므로 단독으로는 안 쓴다.
        if left_x is not None and left_src == 'fit':
            return left_x + self.lane_width / 2.0, left_src
        return None, None
