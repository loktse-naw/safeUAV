from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-18:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + fraction * segment
    return float(np.linalg.norm(point - projection))


def segment_clear_of_circle(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
    epsilon: float = 0.0,
) -> bool:
    return point_segment_distance(center, start, end) >= radius - epsilon


def point_in_bounds(point: np.ndarray, area: Iterable[float], epsilon: float = 0.0) -> bool:
    xmin, xmax, ymin, ymax = [float(value) for value in area]
    return bool(
        xmin - epsilon <= point[0] <= xmax + epsilon
        and ymin - epsilon <= point[1] <= ymax + epsilon
    )


def segment_in_bounds(start: np.ndarray, end: np.ndarray, area: Iterable[float], epsilon: float = 0.0) -> bool:
    return point_in_bounds(start, area, epsilon) and point_in_bounds(end, area, epsilon)


def _tangent_angles(point: np.ndarray, center: np.ndarray, radius: float) -> list[float]:
    relative = point - center
    distance = float(np.linalg.norm(relative))
    if distance <= radius:
        raise ValueError("Safe path endpoint lies inside the expanded no-fly zone")
    theta = math.atan2(relative[1], relative[0])
    alpha = math.acos(radius / distance)
    return [theta - alpha, theta + alpha]


def _positive_angle(angle: float) -> float:
    return angle % (2.0 * math.pi)


def _arc_points(
    center: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    direction: int,
    max_segment_m: float,
) -> tuple[list[np.ndarray], float]:
    if direction > 0:
        delta = _positive_angle(end_angle - start_angle)
    else:
        delta = _positive_angle(start_angle - end_angle)
    arc_length = radius * delta
    segments = max(1, int(math.ceil(arc_length / max_segment_m)))
    angles = [start_angle + direction * delta * idx / segments for idx in range(1, segments + 1)]
    points = [center + radius * np.array([math.cos(angle), math.sin(angle)]) for angle in angles]
    return points, arc_length


def shortest_safe_polyline(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
    arc_resolution_m: float = 4.0,
    epsilon_m: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Shortest path around one expanded circular NFZ.

    The returned polyline uses a radius enlarged by epsilon_m so numerical line
    checks remain outside the forbidden boundary.
    """

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    effective_radius = float(radius + epsilon_m)
    if segment_clear_of_circle(start, end, center, effective_radius):
        return np.vstack([start, end]), float(np.linalg.norm(end - start))

    start_angles = _tangent_angles(start, center, effective_radius)
    end_angles = _tangent_angles(end, center, effective_radius)
    candidates: list[tuple[float, np.ndarray]] = []
    for angle_start in start_angles:
        tangent_start = center + effective_radius * np.array(
            [math.cos(angle_start), math.sin(angle_start)]
        )
        for angle_end in end_angles:
            tangent_end = center + effective_radius * np.array(
                [math.cos(angle_end), math.sin(angle_end)]
            )
            for direction in (-1, 1):
                arc, arc_length = _arc_points(
                    center,
                    effective_radius,
                    angle_start,
                    angle_end,
                    direction,
                    arc_resolution_m,
                )
                points = np.vstack([start, tangent_start, *arc, end])
                if not segment_clear_of_circle(start, tangent_start, center, effective_radius, 1e-7):
                    continue
                if not segment_clear_of_circle(tangent_end, end, center, effective_radius, 1e-7):
                    continue
                length = (
                    float(np.linalg.norm(tangent_start - start))
                    + arc_length
                    + float(np.linalg.norm(end - tangent_end))
                )
                candidates.append((length, points))
    if not candidates:
        raise ValueError("No safe tangent path found around circular NFZ")
    return min(candidates, key=lambda item: item[0])[1], min(item[0] for item in candidates)


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, nodes: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if nodes < 2:
        raise ValueError("nodes must be at least 2")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    targets = np.linspace(0.0, cumulative[-1], nodes)
    output = np.empty((nodes, 2), dtype=np.float64)
    segment = 0
    for idx, target in enumerate(targets):
        while segment < len(distances) - 1 and target > cumulative[segment + 1]:
            segment += 1
        length = distances[segment]
        fraction = 0.0 if length <= 1e-15 else (target - cumulative[segment]) / length
        output[idx] = points[segment] + fraction * (points[segment + 1] - points[segment])
    output[0] = points[0]
    output[-1] = points[-1]
    return output


def advance_along_polyline(points: np.ndarray, distance_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    remaining = max(0.0, float(distance_m))
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        if length <= 1e-15:
            continue
        if remaining <= length:
            return start + remaining / length * (end - start)
        remaining -= length
    return points[-1].copy()

