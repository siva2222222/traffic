"""Camera-motion compensation. Tracks background features frame-to-frame and keeps a transform M that maps
CURRENT-frame pixels -> REFERENCE-frame pixels (the frame your lane zones were drawn on).
Fixed CCTV: M stays identity (dead-band). Shaky / panning / zooming footage: zones and vehicle motion stay correct."""
import cv2
import numpy as np

class CameraStabilizer:
    def __init__(self, scale=0.5):
        self.scale = scale
        self.reset()

    def reset(self):
        self.prev = None
        self.M = np.eye(3)

    def update(self, frame_bgr):
        g = cv2.cvtColor(cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if self.prev is not None and self.prev.shape != g.shape:
            self.prev = None        # source resolution changed mid-stream — restart tracking, don't crash
        if self.prev is not None:
            pts = cv2.goodFeaturesToTrack(self.prev, 300, 0.01, 8)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, pts, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 12:
                    A, inl = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                    scale = np.hypot(A[0, 0], A[1, 0]) if A is not None else 0
                    valid = A is not None and inl is not None and inl.sum() >= 10 \
                        and np.all(np.isfinite(A)) and 0.5 < scale < 2.0     # reject degenerate/runaway estimates
                    if valid:
                        moved = np.hypot(A[0, 2], A[1, 2]) > 0.25 or abs(scale - 1) > 0.0008 \
                            or abs(A[1, 0]) > 0.0008
                        if moved:                                  # dead-band: ignore sub-pixel noise on fixed cameras
                            candidate = self.M @ np.vstack([A, [0, 0, 1]])
                            if np.all(np.isfinite(candidate)) and abs(np.linalg.det(candidate)) > 1e-6:
                                self.M = candidate
        self.prev = g
        M = self.M.copy()
        M[:2, 2] /= self.scale                                     # translation back to full-resolution pixels
        return M

def apply(M, pt):
    v = M @ np.array([pt[0], pt[1], 1.0])
    return float(v[0]), float(v[1])

def apply_poly(M, poly):
    p = np.hstack([poly.reshape(-1, 2).astype(np.float64), np.ones((len(poly), 1))]) @ M.T
    xy = np.nan_to_num(p[:, :2], nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(xy, -2 ** 30, 2 ** 30).astype(np.int32)
