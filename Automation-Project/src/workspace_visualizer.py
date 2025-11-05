#!/usr/bin/env python3
"""
Subie workstation visualizer styled after the reference layout.

The script listens to the SMACH introspection topic published by `scheduler.py`
and renders a schematic workbench with highlighted areas, the current robot
state, the upcoming state, and a simple progress indicator.

Usage
-----
Run the scheduler stack first, then:
    $ python workspace_visualizer.py

Optional: if `shared_state.py` is available in the same directory the visualizer
will use `get_ready_for_next_step()` to colour the “Ready” indicator.
"""

import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import rospy
    from smach_msgs.msg import SmachContainerStatus
except ImportError as exc:  # pragma: no cover - executed when ROS is missing
    raise SystemExit(
        "❌ workspace_visualizer requires ROS (rospy + smach_msgs). "
        "Please ensure you are inside a ROS environment."
    ) from exc

try:
    from shared_state import get_ready_for_next_step  # type: ignore[attr-defined]
except ImportError:
    get_ready_for_next_step = None  # type: ignore[assignment]


# Canvas configuration
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Workbench boundary (top-left x/y, bottom-right x/y)
WORKBENCH_RECT = (60, 60, 1040, 660)
BOUNDARY_COLOR = (0, 255, 0)  # green

# Zone layout: rectangles defined as (x1, y1, x2, y2)
ZONE_LAYOUT: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "Battery": {"rect": (100, 90, 620, 230), "label": "Battery"},
    "Motor": {"rect": (660, 90, 860, 260), "label": "Motor"},
    "MotorTray": {"rect": (660, 90, 860, 260), "label": "Motor"},
    "Assembly": {"rect": (320, 360, 620, 540), "label": "Assembly"},
    "PCB2": {"rect": (320, 260, 620, 320), "label": "PCB2"},
    "PCB1": {"rect": (660, 260, 880, 320), "label": "PCB1"},
    "PCB3": {"rect": (100, 360, 280, 540), "label": "PCB3"},
    "Handover": {"rect": (100, 560, 280, 640), "label": "Handover"},
    "Home": {"rect": (480, 420, 520, 460), "label": "Home"},
}

# Compute rectangle centres and store them for later use
for zone_cfg in ZONE_LAYOUT.values():
    x1, y1, x2, y2 = zone_cfg["rect"]
    zone_cfg["center"] = ((x1 + x2) // 2, (y1 + y2) // 2)

# Target circle inside the handover area (center x, center y, radius)
TARGET_CIRCLE = (190, 620, 30)

# Robot base indicator
ROBOT_BASE_POINT = (600, 420)

# Map SMACH states to their primary zones (used for colouring)
STATE_PRIMARY_ZONE = {
    "Start": "Motor",
    "MPickUp": "Motor",
    "MHold": "Handover",
    "MHoldHD": "Handover",
    "MPositioning": "Assembly",
    "PCB1PickUpAndPositioning": "PCB1",
    "PCB2PickUpAndPositioning": "PCB2",
    "PCB3PickUpAndPositioning": "PCB3",
    "BatteryPickUpAndPositioning": "Battery",
    "Test": "Motor",
    "Aborted": "Handover",
    "finished": "Home",
}

# Define the path (sequence of zone names) that the robot follows per state.
STATE_PATH_TEMPLATE: Dict[str, Sequence[str]] = {
    "Start": ("Home",),
    "MPickUp": ("Motor", "Assembly"),
    "MHold": ("Handover", "Assembly"),
    "MHoldHD": ("Handover", "Assembly"),
    "MPositioning": ("Assembly",),
    "PCB1PickUpAndPositioning": ("Assembly", "PCB1"),
    "PCB2PickUpAndPositioning": ("Assembly", "PCB2"),
    "PCB3PickUpAndPositioning": ("Assembly", "PCB3"),
    "BatteryPickUpAndPositioning": ("Assembly", "Battery"),
    "Test": ("Motor",),
    "Aborted": ("Handover", "Home"),
    "finished": ("Home",),
}


def build_state_paths() -> Dict[str, List[Tuple[int, int]]]:
    """Convert zone sequences to pixel coordinates."""
    paths: Dict[str, List[Tuple[int, int]]] = {}
    for state, zones in STATE_PATH_TEMPLATE.items():
        points: List[Tuple[int, int]] = []
        for zone in zones:
            cfg = ZONE_LAYOUT.get(zone)
            if not cfg:
                continue
            points.append(cfg["center"])  # type: ignore[index]
        if points:
            paths[state] = points
    return paths


STATE_PATHS = build_state_paths()

# Possible outgoing transitions (used to display the most likely next state)
STATE_TRANSITIONS: Dict[str, List[str]] = {
    "Start": ["MPickUp", "MHoldHD", "MPositioning", "PCB1PickUpAndPositioning"],
    "MPickUp": ["MHold", "MHoldHD", "Aborted"],
    "MHold": ["MPositioning", "Aborted"],
    "MHoldHD": ["MPositioning", "finished", "Aborted"],
    "MPositioning": [
        "MPickUp",
        "PCB1PickUpAndPositioning",
        "PCB2PickUpAndPositioning",
        "BatteryPickUpAndPositioning",
    ],
    "PCB1PickUpAndPositioning": ["PCB2PickUpAndPositioning", "Aborted"],
    "PCB2PickUpAndPositioning": ["PCB3PickUpAndPositioning", "Aborted"],
    "PCB3PickUpAndPositioning": ["BatteryPickUpAndPositioning", "Aborted"],
    "BatteryPickUpAndPositioning": ["finished", "MPickUp", "Aborted"],
    "Test": ["finished", "Aborted"],
    "Aborted": ["Start", "MPickUp", "finished"],
    "finished": ["Start"],
}


def get_state_label(name: str) -> str:
    """Pretty-print helper for state names."""
    return name.replace("_", " ")


class SchedulerVisualizer:
    """Render the workbench layout with state overlays."""

    def __init__(self) -> None:
        try:
            rospy.init_node("scheduler_visualizer", anonymous=True)
        except rospy.ROSException as exc:
            raise SystemExit(
                "❌ Could not initialise ROS node. Please start roscore and the scheduler first."
            ) from exc

        self.current_state = "Start"
        self.next_states: List[str] = []
        self.state_since = time.time()
        self.state_history: Deque[str] = deque(maxlen=10)
        self.state_history.append(self.current_state)
        self._lock = threading.Lock()

        rospy.Subscriber(
            "server_name/smach/container_status", SmachContainerStatus, self._status_callback
        )

        cv2.namedWindow("Subie Workbench Overview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Subie Workbench Overview", CANVAS_WIDTH, CANVAS_HEIGHT)

    def _status_callback(self, msg: SmachContainerStatus) -> None:
        if not msg.active_states:
            return

        state_name = msg.active_states[-1]
        with self._lock:
            if state_name != self.current_state:
                self.state_since = time.time()
                self.state_history.append(state_name)
            self.current_state = state_name
            self.next_states = STATE_TRANSITIONS.get(state_name, [])

    # === Drawing helpers ==================================================

    def _draw_workbench(self, canvas: np.ndarray, current_state: str, next_state: Optional[str]) -> None:
        canvas[:] = 255  # white background

        x1, y1, x2, y2 = WORKBENCH_RECT
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOUNDARY_COLOR, thickness=4)
        cv2.putText(canvas, "Bound", (x1, y2 + 18), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        active_zone = STATE_PRIMARY_ZONE.get(current_state)
        next_zone = STATE_PRIMARY_ZONE.get(next_state) if next_state else None

        # Draw schematic zones
        for name, data in ZONE_LAYOUT.items():
            rect = data["rect"]
            label = data.get("label", name)
            colour = (0, 255, 0)
            thickness = 3

            if name == active_zone:
                colour = (0, 0, 255)
                thickness = 5
            elif name == next_zone:
                colour = (0, 165, 255)

            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), colour, thickness)
            cv2.putText(
                canvas,
                label,
                (rect[0] + 6, rect[1] + 20),
                FONT,
                0.6,
                colour if name == active_zone else (0, 150, 0),
                2 if name == active_zone else 1,
                cv2.LINE_AA,
            )

        # Target circle inside handover zone
        cx, cy, radius = TARGET_CIRCLE
        cv2.circle(canvas, (cx, cy), radius, (0, 120, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, "TARGET", (cx - 38, cy - radius - 8), FONT, 0.55, (0, 120, 0), 2, cv2.LINE_AA)

        # Robot base indicator
        cv2.circle(canvas, ROBOT_BASE_POINT, 8, (0, 0, 255), -1)

        # Active path polyline
        points = STATE_PATHS.get(current_state, [])
        if len(points) >= 2:
            self._draw_gradient_path(canvas, points)

    @staticmethod
    def _draw_gradient_path(canvas: np.ndarray, points: Sequence[Tuple[int, int]]) -> None:
        start_color = np.array((0, 0, 255))
        end_color = np.array((0, 255, 0))

        segments = len(points) - 1
        for idx in range(segments):
            p1 = points[idx]
            p2 = points[idx + 1]
            t = idx / max(segments - 1, 1)
            color_vec = (1 - t) * start_color + t * end_color
            color = tuple(int(c) for c in color_vec)
            cv2.line(canvas, p1, p2, color, 6, cv2.LINE_AA)

        cv2.circle(canvas, points[0], 6, (0, 0, 0), -1)
        cv2.circle(canvas, points[-1], 6, (0, 0, 0), -1)

    def _draw_status_panel(
        self, canvas: np.ndarray, current_state: str, upcoming_state: Optional[str], elapsed: float
    ) -> None:
        ready = False
        if callable(get_ready_for_next_step):
            try:
                ready = bool(get_ready_for_next_step())  # type: ignore[misc]
            except Exception:
                ready = False

        # Ready indicator
        ready_rect = (1060, 80, 1220, 130)
        cv2.rectangle(canvas, (ready_rect[0], ready_rect[1]), (ready_rect[2], ready_rect[3]), (0, 0, 0), 2)
        cv2.putText(canvas, "Ready:", (ready_rect[0] + 10, ready_rect[1] + 30), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        indicator_color = (0, 200, 0) if ready else (0, 0, 255)
        cv2.circle(canvas, (ready_rect[2] - 30, ready_rect[1] + 25), 16, indicator_color, -1, cv2.LINE_AA)

        # Current image placeholder / label
        panel_rect = (1060, 160, 1220, 320)
        cv2.rectangle(canvas, (panel_rect[0], panel_rect[1]), (panel_rect[2], panel_rect[3]), (60, 60, 60), 2)
        cv2.rectangle(canvas, (panel_rect[0], panel_rect[1]), (panel_rect[2], panel_rect[3]), (180, 180, 180), -1)

        cv2.putText(canvas, "Current", (panel_rect[0] + 10, panel_rect[1] + 25), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            get_state_label(current_state),
            (panel_rect[0] + 12, panel_rect[1] + 95),
            FONT,
            0.55,
            (150, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Next:",
            (panel_rect[0] + 10, panel_rect[3] + 35),
            FONT,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        next_label = get_state_label(upcoming_state) if upcoming_state else "—"
        cv2.putText(
            canvas,
            next_label,
            (panel_rect[0] + 70, panel_rect[3] + 35),
            FONT,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Progress circle
        centre = (1135, 540)
        radius = 60
        cv2.circle(canvas, centre, radius, (0, 200, 0), 4, cv2.LINE_AA)

        progress = min(elapsed / 30.0, 1.0)  # assume nominal 30s per state
        angle = int(progress * 360)
        cv2.ellipse(canvas, centre, (radius, radius), -90, 0, angle, (0, 200, 0), 8, cv2.LINE_AA)

        elapsed_text = f"{int(elapsed):d}"
        cv2.putText(
            canvas,
            elapsed_text,
            (centre[0] - 15, centre[1] + 8),
            FONT,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def spin(self) -> None:
        rate = rospy.Rate(12)
        try:
            while not rospy.is_shutdown():
                with self._lock:
                    current_state = self.current_state
                    elapsed = time.time() - self.state_since
                    next_state = self.next_states[0] if self.next_states else None

                canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
                self._draw_workbench(canvas, current_state, next_state)
                self._draw_status_panel(canvas, current_state, next_state, elapsed)

                cv2.imshow("Subie Workbench Overview", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    visualizer = SchedulerVisualizer()
    visualizer.spin()


if __name__ == "__main__":
    main()
