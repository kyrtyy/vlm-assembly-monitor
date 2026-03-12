"""
Kalman Filter Tracker
=====================
Bridges the gap between discrete neural network detections (available at ~10 fps)
and the continuous physical reality of assembly components in motion.

This is the key physics-background differentiator in the project.

State vector: x = [cx, cy, w, h, vx, vy, vw, vh]^T
    - (cx, cy): bounding box centre (normalised [0,1])
    - (w, h):   bounding box dimensions (normalised [0,1])
    - (vx, vy, vw, vh): first-order velocities (constant velocity model)

System dynamics (state transition model):
    x_{k+1} = F · x_k + w_k,    w_k ~ N(0, Q)
    
    F = [I₄  dt·I₄]   — constant velocity motion model
        [0₄   I₄  ]

Measurement model:
    z_k = H · x_k + v_k,    v_k ~ N(0, R)
    
    H = [I₄  0₄]   — we only observe (cx, cy, w, h), not velocities

Prediction step (no observation available, e.g. object occluded):
    x̂_{k|k-1} = F · x̂_{k-1|k-1}
    P_{k|k-1}  = F · P_{k-1|k-1} · F^T + Q

Update step (observation z_k available):
    Innovation:  y_k   = z_k - H · x̂_{k|k-1}
    Innovation cov: S_k = H · P_{k|k-1} · H^T + R
    Kalman gain: K_k  = P_{k|k-1} · H^T · S_k^{-1}
    Updated state: x̂_{k|k}  = x̂_{k|k-1} + K_k · y_k
    Updated cov:   P_{k|k}   = (I - K_k · H) · P_{k|k-1}

References:
    Kalman, R. E. (1960). A new approach to linear filtering and prediction problems.
    Welch & Bishop (2006). An introduction to the Kalman Filter. UNC-Chapel Hill TR 95-041.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalmanTracklet:
    """
    Single tracked object with its Kalman state estimate.
    
    Attributes:
        track_id:       Unique identifier.
        state:          x̂  — estimated state vector (8,)
        covariance:     P  — estimated error covariance (8, 8)
        hits:           Number of consecutive successful updates.
        misses:         Number of consecutive missed detections.
        history:        Log of all predicted (cx,cy,w,h) over time.
    """
    track_id: int
    state: np.ndarray              # (8,) [cx, cy, w, h, vx, vy, vw, vh]
    covariance: np.ndarray         # (8, 8)
    hits: int = 0
    misses: int = 0
    history: list = field(default_factory=list)


class BBoxKalmanFilter:
    """
    Multi-object Kalman filter tracker for bounding boxes output by the VLM.
    
    Implements a simple Hungarian-algorithm-based data association to match
    new detections to existing tracklets between frames.
    
    Usage:
        tracker = BBoxKalmanFilter()
        for frame_detections in video:
            tracked_boxes = tracker.update(frame_detections)
    """

    # State dimension: [cx, cy, w, h, vx, vy, vw, vh]
    DIM_X = 8
    # Measurement dimension: [cx, cy, w, h]
    DIM_Z = 4

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 0.01,
        measurement_noise_std: float = 0.05,
        max_misses: int = 5,
        min_hits_to_confirm: int = 2,
        iou_threshold: float = 0.3,
    ):
        self.dt = dt
        self.max_misses = max_misses
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_threshold = iou_threshold
        self._next_id = 0
        self.tracklets: list[KalmanTracklet] = []

        # ── State transition matrix F (constant velocity model) ──────────
        self.F = np.eye(self.DIM_X)
        self.F[:self.DIM_Z, self.DIM_Z:] = dt * np.eye(self.DIM_Z)

        # ── Measurement matrix H ─────────────────────────────────────────
        self.H = np.zeros((self.DIM_Z, self.DIM_X))
        self.H[:self.DIM_Z, :self.DIM_Z] = np.eye(self.DIM_Z)

        # ── Process noise covariance Q ────────────────────────────────────
        # Larger process noise for velocity states (less certain about acceleration)
        q = process_noise_std ** 2
        self.Q = np.diag([q, q, q, q, 4*q, 4*q, 4*q, 4*q])

        # ── Measurement noise covariance R ────────────────────────────────
        r = measurement_noise_std ** 2
        self.R = r * np.eye(self.DIM_Z)

        # ── Initial covariance P₀ ─────────────────────────────────────────
        self.P0 = np.diag([r, r, r, r, 1.0, 1.0, 1.0, 1.0])

    # ──────────────────────────────────────────────────────────────────────
    # Core Kalman operations
    # ──────────────────────────────────────────────────────────────────────

    def _predict(self, track: KalmanTracklet) -> None:
        """Prediction step: propagate state forward by one time step."""
        track.state = self.F @ track.state
        track.covariance = self.F @ track.covariance @ self.F.T + self.Q

    def _update(self, track: KalmanTracklet, z: np.ndarray) -> None:
        """
        Measurement update step given observation z = (cx, cy, w, h).
        
        Solves the linear system S·K^T = (H·P)^T for numerical stability
        (avoids explicit matrix inversion of S).
        """
        S = self.H @ track.covariance @ self.H.T + self.R   # (4, 4)
        K = np.linalg.solve(S.T, (track.covariance @ self.H.T).T).T  # (8, 4)

        innovation = z - self.H @ track.state                         # (4,)
        track.state = track.state + K @ innovation
        I_KH = np.eye(self.DIM_X) - K @ self.H
        # Joseph form of covariance update (numerically stable, preserves symmetry)
        track.covariance = I_KH @ track.covariance @ I_KH.T + K @ self.R @ K.T

    # ──────────────────────────────────────────────────────────────────────
    # Data association (IoU-based Hungarian matching)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """Compute IoU between two (cx, cy, w, h) boxes."""
        ax1, ay1 = boxA[0] - boxA[2] / 2, boxA[1] - boxA[3] / 2
        ax2, ay2 = boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2
        bx1, by1 = boxB[0] - boxB[2] / 2, boxB[1] - boxB[3] / 2
        bx2, by2 = boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2

        inter_x = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_y = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_x * inter_y

        areaA = (ax2 - ax1) * (ay2 - ay1)
        areaB = (bx2 - bx1) * (by2 - by1)
        union = areaA + areaB - inter + 1e-9
        return inter / union

    def _hungarian_match(
        self,
        predictions: np.ndarray,   # (M, 4) predicted (cx,cy,w,h)
        detections: np.ndarray,    # (N, 4) new detections
    ) -> tuple[list[tuple[int,int]], list[int], list[int]]:
        """
        Greedy IoU matching (O(MN) — sufficient for max_objects ≤ 10).
        Returns: (matches, unmatched_tracks, unmatched_detections)
        """
        if len(predictions) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(predictions))), []

        # IoU cost matrix
        iou_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(pred, det)

        # Greedy matching: iterate over highest IoU pairs
        matched_tracks, matched_dets = set(), set()
        matches = []
        flat_idx = np.argsort(-iou_matrix.ravel())
        for idx in flat_idx:
            i, j = divmod(idx, len(detections))
            if i in matched_tracks or j in matched_dets:
                continue
            if iou_matrix[i, j] < self.iou_threshold:
                break
            matches.append((i, j))
            matched_tracks.add(i)
            matched_dets.add(j)

        unmatched_tracks = [i for i in range(len(predictions)) if i not in matched_tracks]
        unmatched_dets   = [j for j in range(len(detections))  if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        detections: np.ndarray,             # (N, 4) or (N, 5) with objectness
        objectness_threshold: float = 0.4,
    ) -> list[dict]:
        """
        Process one frame of detections.

        Args:
            detections: Array of shape (N, 4) — (cx,cy,w,h) — or (N,5) with a
                        fifth column being objectness score. Boxes below 
                        objectness_threshold are filtered out.
            objectness_threshold: Minimum objectness score to accept a detection.

        Returns:
            List of dicts with keys: {track_id, box, hits, confirmed}
            Only confirmed tracklets (hits ≥ min_hits_to_confirm) are returned.
        """
        # Filter by objectness if scores provided
        if detections.shape[1] == 5:
            mask = detections[:, 4] >= objectness_threshold
            detections = detections[mask, :4]

        # Step 1: Predict all existing tracklets forward
        predicted_boxes = []
        for track in self.tracklets:
            self._predict(track)
            predicted_boxes.append(self.H @ track.state)  # (cx,cy,w,h)

        # Step 2: Data association
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.zeros((0, 4))
        matches, unmatched_tracks, unmatched_dets = self._hungarian_match(
            predicted_boxes, detections
        )

        # Step 3: Update matched tracklets
        for track_idx, det_idx in matches:
            self._update(self.tracklets[track_idx], detections[det_idx])
            self.tracklets[track_idx].hits += 1
            self.tracklets[track_idx].misses = 0

        # Step 4: Increment miss counter for unmatched tracklets
        for track_idx in unmatched_tracks:
            self.tracklets[track_idx].misses += 1

        # Step 5: Initialise new tracklets for unmatched detections
        for det_idx in unmatched_dets:
            z = detections[det_idx]
            state = np.zeros(self.DIM_X)
            state[:self.DIM_Z] = z
            new_track = KalmanTracklet(
                track_id=self._next_id,
                state=state,
                covariance=self.P0.copy(),
            )
            self._next_id += 1
            self.tracklets.append(new_track)

        # Step 6: Delete dead tracklets
        self.tracklets = [t for t in self.tracklets if t.misses <= self.max_misses]

        # Step 7: Log history for confirmed tracks
        results = []
        for track in self.tracklets:
            box = (self.H @ track.state).clip(0, 1)  # (cx,cy,w,h) clamped
            track.history.append(box.copy())
            confirmed = track.hits >= self.min_hits_to_confirm
            results.append({
                "track_id":  track.track_id,
                "box":       box,
                "hits":      track.hits,
                "misses":    track.misses,
                "confirmed": confirmed,
            })

        return results

    def reset(self) -> None:
        """Clear all tracklets (call between clips/sequences)."""
        self.tracklets = []
        self._next_id = 0
