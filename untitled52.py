import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from scipy.interpolate import CubicSpline
from scipy.special import fresnel
import pandas as pd

# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (ίδιες όπως untitled50.py)
# =====================================================
def segment_plane_intersection(point_a, point_b, plane):
    segment_dir = point_b - point_a
    segment_length = np.linalg.norm(segment_dir)
    if segment_length < 1e-12:
        return None
    segment_dir = segment_dir / segment_length
    denom = np.dot(segment_dir, plane['normal'])
    if abs(denom) < 1e-12:
        return None
    t = np.dot(plane['point'] - point_a, plane['normal']) / denom
    if 0 <= t <= segment_length:
        return point_a + t * segment_dir
    return None

def intersect_rail_with_plane_analytical(system, plane, s_center, s_window=100.0, num_samples=500):
    s_min = max(s_center - s_window/2, 0)
    s_max = min(s_center + s_window/2, system.path_func['length'])
    s_values = np.linspace(s_min, s_max, num_samples)
    intersection_points_3d = []
    intersection_points_2d = []
    for i, s in enumerate(s_values):
        rail_profile_3d, frame = system.get_rail_profile_at_s(s)
        for j in range(len(rail_profile_3d) - 1):
            point_a = rail_profile_3d[j]
            point_b = rail_profile_3d[j + 1]
            intersection = segment_plane_intersection(point_a, point_b, plane)
            if intersection is not None:
                intersection_points_3d.append(intersection)
                rel = intersection - plane['point']
                x_local = np.dot(rel, plane['x_axis'])
                y_local = np.dot(rel, plane['y_axis'])
                intersection_points_2d.append([x_local, y_local])
    return np.array(intersection_points_3d), np.array(intersection_points_2d)

def create_rail_polygon_from_intersection(points_2d):
    if len(points_2d) == 0:
        return np.array([])
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points_2d)
        ordered_points = points_2d[hull.vertices]
        return np.vstack([ordered_points, ordered_points[0]])
    except:
        return np.vstack([points_2d, points_2d[0]])

def segment_intersection(p1, p2, q1, q2):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-12: return None
    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/denom
    def on_segment(a, b, p): return min(a[0],b[0])-1e-10<=p[0]<=max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10<=p[1]<=max(a[1],b[1])+1e-10
    pt = np.array([px, py])
    if on_segment(p1, p2, pt) and on_segment(q1, q2, pt): return pt
    return None

def wheel_polygon_contact_segments_multi(wheel_2d, rail_poly):
    path = Path(rail_poly)
    n = len(wheel_2d)
    segments = []
    in_mask = path.contains_points(wheel_2d)
    for i in range(n-1):
        A, B = wheel_2d[i], wheel_2d[i+1]
        in_A, in_B = in_mask[i], in_mask[i+1]
        intersections = []
        for j in range(len(rail_poly)-1):
            Q1, Q2 = rail_poly[j], rail_poly[j+1]
            pt = segment_intersection(A, B, Q1, Q2)
            if pt is not None: 
                intersections.append(pt)
        points = []
        if in_A: points.append(A)
        for pt in intersections: points.append(pt)
        if in_B: points.append(B)
        if len(points) >= 2:
            points = np.array(points)
            dists = np.linalg.norm(points - A, axis=1)
            idx = np.argsort(dists)
            sorted_points = points[idx]
            for k in range(len(sorted_points)-1):
                mid = 0.5*(sorted_points[k] + sorted_points[k+1])
                if path.contains_point(mid):
                    segments.append([sorted_points[k], sorted_points[k+1]])
        elif in_A and in_B:
            segments.append([A, B])
    all_pts = []
    for seg in segments:
        all_pts.append(tuple(map(tuple, seg)))
    polylines = []
    while all_pts:
        start, end = all_pts.pop(0)
        poly = [start, end]
        changed = True
        while changed:
            changed = False
            for i, (s, e) in enumerate(all_pts):
                if np.allclose(poly[-1], s):
                    poly.append(e)
                    all_pts.pop(i)
                    changed = True
                    break
                elif np.allclose(poly[-1], e):
                    poly.append(s)
                    all_pts.pop(i)
                    changed = True
                    break
                elif np.allclose(poly[0], s):
                    poly = [e] + poly
                    all_pts.pop(i)
                    changed = True
                    break
                elif np.allclose(poly[0], e):
                    poly = [s] + poly
                    all_pts.pop(i)
                    changed = True
                    break
        polylines.append(np.array(poly))
    return polylines

def softplus(x, beta=100.0):
    return (1.0 / beta) * np.log(1 + np.exp(beta * x))

def project_to_plane_analytical(points, plane):
    rel = points - plane['point']
    x_local = np.dot(rel, plane['x_axis'])
    y_local = np.dot(rel, plane['y_axis'])
    return np.column_stack([x_local, y_local])

def rail_polygon_n_lines(profile_2d):
    return np.vstack([profile_2d, profile_2d[0]])

def split_polyline_by_length(polyline, max_length):
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(seg_lengths)
    if total_length <= max_length:
        return [polyline]
    cum_length = np.cumsum(seg_lengths)
    split_idx = np.searchsorted(cum_length, total_length / 2)
    if split_idx == 0:
        return [polyline]
    prev_len = cum_length[split_idx-1] if split_idx > 0 else 0
    remain = (total_length/2 - prev_len) / seg_lengths[split_idx]
    split_point = polyline[split_idx] * (1-remain) + polyline[split_idx+1] * remain
    poly1 = np.vstack([polyline[:split_idx+1], split_point])
    poly2 = np.vstack([split_point, polyline[split_idx+1:]])
    return [poly1, poly2]

def find_max_min_distance_vector_dense(wheel_poly, rail_poly, samples_per_segment=20):
    sampled_wheel = sample_points_along_polyline(wheel_poly, samples_per_segment)
    seg_a = rail_poly[:-1]
    seg_b = rail_poly[1:]
    pa = sampled_wheel[:, None, :]
    ba = seg_b - seg_a
    ba_len2 = np.sum(ba**2, axis=1)
    ba_len2 = np.where(ba_len2 == 0, 1, ba_len2)
    pa_minus_a = pa - seg_a
    t = np.sum(pa_minus_a * ba, axis=2) / ba_len2
    t = np.clip(t, 0, 1)
    projections = seg_a + t[..., None]*ba
    dists = np.linalg.norm(pa - projections, axis=2)
    min_idx = np.argmin(dists, axis=1)
    min_distances = dists[np.arange(len(sampled_wheel)), min_idx]
    closest_rail_pts = projections[np.arange(len(sampled_wheel)), min_idx]
    vectors = closest_rail_pts - sampled_wheel
    max_idx = np.argmax(min_distances)
    max_min_dist = min_distances[max_idx]
    contact_vec = vectors[max_idx]
    wheel_pt = sampled_wheel[max_idx]
    rail_pt = closest_rail_pts[max_idx]
    return max_min_dist, contact_vec, wheel_pt, rail_pt

def sample_points_along_polyline(polyline, samples_per_segment=20):
    points = []
    for i in range(len(polyline)-1):
        p0 = polyline[i]
        p1 = polyline[i+1]
        for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
            pt = (1-t)*p0 + t*p1
            points.append(pt)
    points.append(polyline[-1])
    return np.array(points)

def compute_friction_coefficient(v_tan_norm, mu_dynamic=0.2, v_d=1e-3):
    safe_vd = max(v_d, 1e-12)
    clipped_v = max(v_tan_norm, 0.0)
    return mu_dynamic * min(clipped_v / safe_vd, 1.0)

# =====================================================
# ANALYTICAL RAIL PATH 3D (ΓΕΩΜΕΤΡΙΑ ΣΤΡΟΦΗΣ ΜΟΝΟ)
# =====================================================
class AnalyticalRailPath3D:
    """
    Path για στροφή με clothoid και banking σύμφωνα με τις προδιαγραφές:
    - Και οι δύο ράγες ξεκινούν με 20m ευθεία
    - Δεξιά: clothoid χωρίς ανύψωση
    - Αριστερή: clothoid με σταδιακή ανύψωση μέχρι 100mm στα 40m
    Άξονες: x=αριστερά, y=πάνω, z=μπροστά.
    """
    def __init__(self, 
                 wheel_type="right",
                 R_base=5000000.0,      # Βασική ακτίνα στροφής
                 gauge=1531.374,        # Απόσταση ράγας
                 h=100.0,               # Μέγιστη ανύψωση (mm)
                 L_straight=20000.0,    # 20m ευθεία (mm)
                 L_incline=20000.0,     # 20m clothoid με banking (mm)
                 L_clothoid_rest=300000.0,  # υπόλοιπο clothoid (mm)
                 E=0.12, 
                 Ud=0.075):
        self.wheel_type = wheel_type.lower()
        self.R_base = R_base
        self.gauge = gauge
        self.h = h
        self.L_straight = L_straight
        self.L_incline = L_incline
        self.L_clothoid_rest = L_clothoid_rest
        self.E = E
        self.Ud = Ud
        self.L_clothoid_total = L_incline + L_clothoid_rest
        self.s_total = L_straight + self.L_clothoid_total
        self._precompute_path()
    def _precompute_path(self):
        if self.wheel_type == "left":
            x_shift = self.gauge/2
            R = self.R_base + self.gauge/2
            self.shift = np.array([x_shift, 140, 0])
            z1 = np.linspace(0, self.L_straight, 200)
            x1 = np.zeros_like(z1)
            y1 = np.zeros_like(z1)
            r1 = np.column_stack([x1, y1, z1])
            A = np.sqrt(R * self.L_clothoid_total)
            s2 = np.linspace(0, self.L_clothoid_total, 600)
            u2 = s2 / A
            C2, S2 = fresnel(u2)
            x2 = -A * C2
            z2 = A * S2
            x2 = x2 - x2[0]
            z2 = z2 - z2[0] + self.L_straight
            y2 = np.zeros_like(x2)
            incline_mask = s2 <= self.L_incline
            y2[incline_mask] = np.interp(s2[incline_mask], [0, self.L_incline], [0, self.h])
            y2[~incline_mask] = self.h
            r2 = np.column_stack([x2, y2, z2])
            self.path_points = np.vstack([r1, r2]) + self.shift
        else:  # right
            x_shift = -self.gauge/2
            R = self.R_base - self.gauge/2
            self.shift = np.array([x_shift, 140, 0])
            z1 = np.linspace(0, self.L_straight, 200)
            x1 = np.zeros_like(z1)
            y1 = np.zeros_like(z1)
            r1 = np.column_stack([x1, y1, z1])
            A = np.sqrt(R * self.L_clothoid_total)
            s2 = np.linspace(0, self.L_clothoid_total, 600)
            u2 = s2 / A
            C2, S2 = fresnel(u2)
            x2 = -A * C2
            z2 = A * S2
            x2 = x2 - x2[0]
            z2 = z2 - z2[0] + self.L_straight
            y2 = np.zeros_like(x2)
            r2 = np.column_stack([x2, y2, z2])
            self.path_points = np.vstack([r1, r2]) + self.shift
        self._create_splines()
    def _create_splines(self):
        s_values = np.linspace(0, self.s_total, len(self.path_points))
        self.x_spline = CubicSpline(s_values, self.path_points[:, 0])
        self.y_spline = CubicSpline(s_values, self.path_points[:, 1])
        self.z_spline = CubicSpline(s_values, self.path_points[:, 2])
        self.dx_spline = self.x_spline.derivative()
        self.dy_spline = self.y_spline.derivative()
        self.dz_spline = self.z_spline.derivative()
    def position(self, s):
        s = np.array(s, ndmin=1, dtype=float)
        s_clipped = np.clip(s, 0, self.s_total)
        x = self.x_spline(s_clipped)
        y = self.y_spline(s_clipped)
        z = self.z_spline(s_clipped)
        return x, y, z
    def tangent(self, s):
        s = np.array(s, ndmin=1, dtype=float)
        s_clipped = np.clip(s, 0, self.s_total)
        dx = self.dx_spline(s_clipped)
        dy = self.dy_spline(s_clipped)
        dz = self.dz_spline(s_clipped)
        norms = np.sqrt(dx**2 + dy**2 + dz**2)
        valid_norms = norms > 1e-12
        dx[valid_norms] /= norms[valid_norms]
        dy[valid_norms] /= norms[valid_norms]
        dz[valid_norms] /= norms[valid_norms]
        dx[~valid_norms] = 0
        dy[~valid_norms] = 0
        dz[~valid_norms] = 1
        if np.isscalar(s):
            return dx[0], dy[0], dz[0]
        return dx, dy, dz
    def curvature(self, s):
        s = np.array(s, ndmin=1, dtype=float)
        s_clipped = np.clip(s, 0, self.s_total)
        kappa = np.zeros_like(s_clipped)
        eps = 1.0
        for i, ss in enumerate(s_clipped):
            s_plus = min(ss + eps, self.s_total)
            s_minus = max(ss - eps, 0)
            t_plus = np.array(self.tangent(s_plus))
            t_minus = np.array(self.tangent(s_minus))
            dT_ds = (t_plus - t_minus) / (2 * eps)
            kappa[i] = np.linalg.norm(dT_ds)
        return kappa
    def superelevation(self, s):
        s = np.array(s, ndmin=1, dtype=float)
        s_clipped = np.clip(s, 0, self.s_total)
        if self.wheel_type == "left":
            max_banking = np.arctan(self.h / self.gauge)
            banking = np.zeros_like(s_clipped)
            incline_mask = s_clipped <= (self.L_straight + self.L_incline)
            straight_mask = s_clipped <= self.L_straight
            banking[straight_mask] = 0
            clothoid_mask = incline_mask & ~straight_mask
            s_clothoid = s_clipped[clothoid_mask] - self.L_straight
            banking[clothoid_mask] = np.interp(s_clothoid, [0, self.L_incline], [0, max_banking])
            banking[~incline_mask] = max_banking
            return banking
        else:
            return np.zeros_like(s_clipped)

# =====================================================
# WheelRailSystemAnalytical ΚΑΙ ΟΛΑ ΤΑ ΥΠΟΛΟΙΠΑ (ίδια όπως untitled50.py)
# =====================================================
class WheelRailSystemAnalytical:
    def __init__(self, wheel_center, wheel_quat, V, omega, wheel_type="right"):
        self.wheel_center = np.array(wheel_center)
        self.V = np.array(V)
        self.omega = np.array(omega)
        self.wheel_quat = np.array(wheel_quat)
        self.wheel_type = wheel_type
        self.rot = R.from_quat(wheel_quat)
        self.WHEEL_POINTS_X = -np.array([39.959, 33.543, 30.832, 25.252,
                                        15.553, -68.447, -71.645, -75.008, -75.947]) + 7.943
        self.WHEEL_POINTS_R = np.abs(np.array([-475.088, -469.879, -462.389,
                                               -453.968, -451.327, -447.130, -446.470, -444.859, -442.370]))
        self.RAIL_POINTS_X = np.array([-33.632, -34.622, -35, -35, -31.523, -23, 0,
                                       23, 31.523, 35, 35, 34.622, 33.632])
        self.RAIL_POINTS_Y = np.array([109.244, 109.972, 111.142, 126.2, 135.002, 139.114,
                                       140, 139.114, 135.002, 126.2, 111.142, 109.972, 109.244]) - 140
        self.create_extended_analytical_path()
    def create_extended_analytical_path(self):
        # Τα παρακάτω H/Lvert/s0 ισχύουν μόνο αν θες bump, αλλά κρατάμε γεωμετρία στροφής
        self.analytical_path = AnalyticalRailPath3D(
            wheel_type=self.wheel_type,
            R_base=5000000.0,
            gauge=1531.374,
            h=100.0,
            L_straight=20000.0,
            L_incline=20000.0,
            L_clothoid_rest=300000.0,
            E=0.12, Ud=0.075
        )
        self.path_func = {
            'length': self.analytical_path.s_total,
            'gauge': 1531.374
        }
        self.x_shift = self.path_func['gauge']/2.0 if self.wheel_type == "left" else -self.path_func['gauge']/2.0
        self.y_shift = 140.0
    def get_path_point(self, s):
        x, y, z = self.analytical_path.position(s)
        return np.array([x[0] + self.x_shift, y[0] + self.y_shift, z[0]])
    def get_frenet_frame(self, s):
        point = self.get_path_point(s)
        tangent = np.array(self.analytical_path.tangent(s))
        h = 1.0
        s_plus = min(s + h, self.path_func['length'])
        s_minus = max(s - h, 0)
        point_plus = self.get_path_point(s_plus)
        point_minus = self.get_path_point(s_minus)
        second_derivative = (point_plus - 2*point + point_minus) / (h**2)
        if np.linalg.norm(second_derivative) > 1e-10:
            normal = second_derivative / np.linalg.norm(second_derivative)
        else:
            normal = np.array([0, 1, 0])
        normal = normal - np.dot(normal, tangent) * tangent
        if np.linalg.norm(normal) > 1e-10:
            normal /= np.linalg.norm(normal)
        else:
            normal = np.array([0, 1, 0])
        tangent = tangent.flatten()[:3]
        normal = normal.flatten()[:3]
        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal)
        normal = np.cross(binormal, tangent)
        normal /= np.linalg.norm(normal)
        return {
            'origin': point,
            'tangent': tangent,
            'normal': normal,
            'binormal': binormal
        }
    def get_first_plane_analytical(self):
        wheel_z = self.wheel_center[2]
        s_guess = wheel_z
        frame = self.get_frenet_frame(s_guess)
        x_axis = self.rot.apply([1, 0, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        normal = frame['normal']
        projection = np.dot(normal, x_axis) * x_axis
        normal_corrected = normal - projection
        if np.linalg.norm(normal_corrected) < 1e-12:
            normal_corrected = np.array([0, 1, 0])
        normal_corrected = normal_corrected / np.linalg.norm(normal_corrected)
        if normal_corrected[1] < 0:
            normal_corrected = -normal_corrected
        y_axis = normal_corrected
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        if z_axis[2] < 0:
            z_axis = -z_axis
            y_axis = np.cross(z_axis, x_axis)
        contact_plane = {
            'point': self.wheel_center,
            'normal': z_axis,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'frenet_frame': frame,
            's_position': s_guess
        }
        return contact_plane
    def get_rail_profile_at_s(self, s):
        frame = self.get_frenet_frame(s)
        tangent = frame['tangent']
        normal = frame['normal']
        binormal = frame['binormal']
        if normal[1] < 0:
            normal = -normal
            binormal = np.cross(tangent, normal)
        if self.wheel_type == "left":
            if binormal[0] > 0:
                binormal = -binormal
                normal = np.cross(binormal, tangent)
        else:
            if binormal[0] > 0:
                binormal = -binormal
                normal = np.cross(binormal, tangent)
        rail_points_3d = []
        for i in range(len(self.RAIL_POINTS_X)):
            point_3d = (frame['origin'] + 
                       self.RAIL_POINTS_X[i] * binormal + 
                       self.RAIL_POINTS_Y[i] * normal)
            rail_points_3d.append(point_3d)
        return np.array(rail_points_3d), {'tangent': tangent, 'normal': normal, 'binormal': binormal, 'origin': frame['origin']}
    # Τα υπόλοιπα όπως untitled50.py (twist, κλπ)
    def set_twist_table(self, twist_points, degrees=False):
        arr = np.array(sorted(twist_points, key=lambda x: x[0]), dtype=float)
        if arr.size == 0:
            self.twist_s = np.array([0.0])
            self.twist_theta = np.array([0.0])
        else:
            self.twist_s = arr[:, 0]
            self.twist_theta = arr[:, 1]
            if degrees:
                self.twist_theta = np.deg2rad(self.twist_theta)
    def get_twist_angle(self, s):
        if not hasattr(self, 'twist_s'):
            return 0.0
        return float(np.interp(s, self.twist_s, self.twist_theta, left=self.twist_theta[0], right=self.twist_theta[-1]))
    def get_twisted_frame(self, s):
        frame = self.get_frenet_frame(s)
        origin = frame['origin']
        tangent = np.array(frame['tangent'], dtype=float)
        tnorm = np.linalg.norm(tangent)
        if tnorm < 1e-12:
            tangent = np.array([0., 1., 0.])
            tnorm = 1.0
        tangent = tangent / tnorm
        normal = np.array(frame['normal'], dtype=float)
        binormal = np.array(frame['binormal'], dtype=float)
        theta = self.get_twist_angle(s)
        rot = R.from_rotvec(tangent * theta)
        normal_tw = rot.apply(normal)
        binormal_tw = rot.apply(binormal)
        normal_tw = normal_tw - np.dot(normal_tw, tangent) * tangent
        if np.linalg.norm(normal_tw) < 1e-12:
            normal_tw = np.array([0., 1., 0.])
            normal_tw = normal_tw - np.dot(normal_tw, tangent) * tangent
        normal_tw /= np.linalg.norm(normal_tw)
        binormal_tw = np.cross(tangent, normal_tw)
        binormal_tw /= np.linalg.norm(binormal_tw)
        return {
            'origin': origin,
            'tangent': tangent,
            'normal': normal_tw,
            'binormal': binormal_tw
        }
    def get_rail_profile_at_s_with_twist(self, s):
        frame = self.get_twisted_frame(s)
        tangent = frame['tangent']
        normal = frame['normal']
        binormal = frame['binormal']
        origin = frame['origin']
        rail_points_3d = []
        for i in range(len(self.RAIL_POINTS_X)):
            point_3d = origin + self.RAIL_POINTS_X[i] * binormal + self.RAIL_POINTS_Y[i] * normal
            rail_points_3d.append(point_3d)
        rail_points_3d = np.array(rail_points_3d)
        return rail_points_3d, {'tangent': tangent, 'normal': normal, 'binormal': binormal, 'origin': origin}

class RailTwistFunction:
    def __init__(self, s_points, theta_points, degrees=False):
        self.s_points = np.array(s_points)
        self.theta_points = np.array(theta_points)
        if degrees:
            self.theta_points = np.deg2rad(self.theta_points)
    def twist(self, s):
        return float(np.interp(s, self.s_points, self.theta_points, left=self.theta_points[0], right=self.theta_points[-1]))

# Οι υπόλοιπες συναρτήσεις, GFOSUB, plot, export, όπως untitled50.py
def sweep_profile_point_3d(profile_point_2d, s, path_func, twist_func):
    origin, tangent, normal, binormal = path_func(s)
    twist_angle = twist_func.twist(s)
    rot = R.from_rotvec(tangent * twist_angle)
    normal_tw = rot.apply(normal)
    binormal_tw = rot.apply(binormal)
    if normal_tw[1] < 0:
        normal_tw = -normal_tw
        binormal_tw = np.cross(tangent, normal_tw)
        binormal_tw /= np.linalg.norm(binormal_tw)
    x, y = profile_point_2d
    pt_3d = origin + x * binormal_tw + y * normal_tw
    return pt_3d

def sweep_full_profile_3d(profile_2d, s, path_func, twist_func):
    return np.array([
        sweep_profile_point_3d(pt, s, path_func, twist_func)
        for pt in profile_2d
    ])

def compute_profile_paths(profile_2d, s_values, path_func, twist_func):
    M = profile_2d.shape[0]
    N = len(s_values)
    paths = np.zeros((M, N, 3))
    for i, pt in enumerate(profile_2d):
        for j, s in enumerate(s_values):
            paths[i, j] = sweep_profile_point_3d(pt, s, path_func, twist_func)
    return paths

def find_intersections_with_plane(profile_paths, plane):
    M, N, _ = profile_paths.shape
    intersection_points_3d = []
    intersection_points_2d = []
    for i in range(M):
        for j in range(N-1):
            pt_a = profile_paths[i, j]
            pt_b = profile_paths[i, j+1]
            seg_dir = pt_b - pt_a
            denom = np.dot(seg_dir, plane['normal'])
            if abs(denom) < 1e-12:
                continue
            t = np.dot(plane['point'] - pt_a, plane['normal']) / denom
            if 0 <= t <= 1:
                inter_pt = pt_a + t * seg_dir
                intersection_points_3d.append(inter_pt)
                rel = inter_pt - plane['point']
                x_local = np.dot(rel, plane['x_axis'])
                y_local = np.dot(rel, plane['y_axis'])
                intersection_points_2d.append([x_local, y_local])
    return np.array(intersection_points_3d), np.array(intersection_points_2d)

def GFOSUB_enhanced_analytical(wheel_center, V, omega, euler_angles, wheel_type="right", return_plot_data=True):
    rot = R.from_euler('zyx', euler_angles)
    wheel_quat = rot.as_quat()
    system = WheelRailSystemAnalytical(wheel_center, wheel_quat, V, omega, wheel_type)
    twist_func = RailTwistFunction(
        s_points=[0, 50000, 120000],
        theta_points=[0, 10/57.30, 0.0],
        degrees=False
    )
    system.set_twist_table(
        list(zip(twist_func.s_points, twist_func.theta_points)),
        degrees=False
    )
    plane = system.get_first_plane_analytical()
    profile_2d = np.column_stack([system.RAIL_POINTS_X, system.RAIL_POINTS_Y])
    s_center = plane['s_position']
    s_window = 500.0
    num_samples = 10
    s_values = np.linspace(s_center-s_window/2, s_center+s_window/2, num_samples)
    def path_func(s):
        frame = system.get_twisted_frame(s)
        return frame['origin'], frame['tangent'], frame['normal'], frame['binormal']
    profile_paths = compute_profile_paths(profile_2d, s_values, path_func, twist_func)
    rail_intersection_3d, rail_intersection_2d = find_intersections_with_plane(profile_paths, plane)
    rail_poly = create_rail_polygon_from_intersection(rail_intersection_2d)
    sign_x = 1 if wheel_type == 'left' else -1
    wheel_2d = np.column_stack([sign_x * system.WHEEL_POINTS_X, -system.WHEEL_POINTS_R])
    wheel_contact_polylines = wheel_polygon_contact_segments_multi(wheel_2d, rail_poly)
    K = 1000000
    n = 1.5
    C_max = 100
    penetration_limit = 0.1
    max_penetration_allowed = 5
    max_damping_velocity = 10.0 / 0.001
    mu_dynamic = 0.2
    v_threshold = 20
    wheel_center_np = np.array(wheel_center)
    F_total = np.zeros(3)
    M_total = np.zeros(3)
    forces_plot_main = []
    max_length = 15.0
    max_min_info = []
    for i, poly in enumerate(wheel_contact_polylines):
        subpolys = split_polyline_by_length(poly, max_length)
        for j, subpoly in enumerate(subpolys):
            if len(subpoly) > 1:
                max_min_dist, contact_vec, wheel_pt, rail_pt = find_max_min_distance_vector_dense(
                    subpoly, rail_poly, samples_per_segment=30
                )
                max_min_info.append({
                    "max_min_dist": max_min_dist,
                    "vec": contact_vec,
                    "pt_on_poly": wheel_pt,
                    "pt_on_polygon": rail_pt,
                    "axes": (plane['point'], plane['x_axis'], plane['y_axis'], plane['z_axis'])
                })
    for i, info in enumerate(max_min_info):
        penetration_depth = abs(info["max_min_dist"])
        penetration_depth = softplus(penetration_depth, beta=20)
        if penetration_depth > max_penetration_allowed:
            penetration_depth = max_penetration_allowed
        origin, x_axis, y_axis, z_axis = info["axes"]
        Pmax2d = info["pt_on_poly"]
        Qrail2d = info["pt_on_polygon"]
        Pmax = origin + Pmax2d[0] * x_axis + Pmax2d[1] * y_axis
        Qrail = origin + Qrail2d[0] * x_axis + Qrail2d[1] * y_axis
        penetration_dir = Pmax - Qrail
        penetration_dir_norm = np.linalg.norm(penetration_dir)
        if penetration_dir_norm < 1e-12:
            continue
        penetration_dir = penetration_dir / penetration_dir_norm
        Fx = np.dot(penetration_dir, x_axis)
        Fy = np.dot(penetration_dir, y_axis)
        Fz = np.dot(penetration_dir, z_axis)
        penetration_dir_mod = Fx * x_axis + Fy * y_axis + Fz * z_axis
        penetration_dir_mod_norm = np.linalg.norm(penetration_dir_mod)
        if penetration_dir_mod_norm < 1e-12:
            continue
        penetration_dir_mod = penetration_dir_mod / penetration_dir_mod_norm
        F_elastic = -K * (penetration_depth ** n) * penetration_dir_mod
        v_point = np.array(V) + np.cross(np.array(omega), (Pmax - wheel_center_np))
        elastic_dir = -penetration_dir_mod
        v_proj_elastic = np.dot(v_point, elastic_dir)
        if penetration_depth > 1e-6 and v_proj_elastic < -1e-6:
            C_effective = C_max * min(penetration_depth / penetration_limit, 1.0)
            v_damp_clipped = min(abs(v_proj_elastic), max_damping_velocity) * np.sign(v_proj_elastic)
            F_damp = C_effective * abs(v_damp_clipped) * elastic_dir
        else:
            F_damp = np.zeros(3)
        F_normal = np.linalg.norm(F_elastic + F_damp)
        v_tan = v_point - np.dot(v_point, penetration_dir_mod) * penetration_dir_mod
        v_tan_norm = np.linalg.norm(v_tan)
        v_tan_safe = max(v_tan_norm, 1e-8)
        tangent_unit_vector = v_tan / v_tan_safe
        mu = compute_friction_coefficient(v_tan_norm, mu_dynamic, v_threshold)
        F_friction = -mu * F_normal * tangent_unit_vector
        r_vec = Pmax - wheel_center_np
        M_elastic = np.cross(r_vec, F_elastic)
        M_damp = np.cross(r_vec, F_damp)
        M_friction = np.cross(r_vec, F_friction)
        F_total += F_elastic + F_damp + F_friction
        M_total += M_elastic + M_damp + M_friction
        forces_plot_main.append({
            'pt': Pmax,
            'F_elastic': F_elastic,
            'F_damp': F_damp,
            'F_friction': F_friction,
            'penetration_dir_mod': penetration_dir_mod,
            'penetration_depth': penetration_depth,
            'tangent_unit_vector': tangent_unit_vector
        })
    if return_plot_data:
        return {
            "system": system,
            "contact_plane": plane,
            "rail_intersection_3d": rail_intersection_3d,
            "rail_intersection_2d": rail_intersection_2d,
            "rail_poly": rail_poly,
            "wheel_2d": wheel_2d,
            "contact_segments": wheel_contact_polylines,
            "contact_forces": forces_plot_main,
            "F_total": F_total,
            "M_total": M_total
        }
    else:
        return list(F_total) + list(M_total)

    
def plot_3d_results_with_forces(results, force_scale=0.01):
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(121, projection='3d')

    system = results["system"]
    s_vals = np.linspace(0, system.path_func['length'], 100)
    path_points = np.array([system.get_path_point(s) for s in s_vals])
    ax1.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 
             'm-', linewidth=3, alpha=0.7, label='Rail Path')

    # Sweeped rail intersection points (και polyline)
    if 'rail_intersection_3d' in results and len(results['rail_intersection_3d']) > 0:
        rail_intersection = results['rail_intersection_3d']
        ax1.scatter(rail_intersection[:, 0], rail_intersection[:, 1], rail_intersection[:, 2],
                   c='blue', s=50, marker='o', label='Rail-Plane Intersection')
    if 'rail_poly' in results and len(results['rail_poly']) > 0:
        # Optional: plot the polyline as swept in 3D (προβολή στο plane)
        pass

    wheel_center = system.wheel_center
    ax1.scatter(wheel_center[0], wheel_center[1], wheel_center[2], c='r', marker='o', s=100, label='Wheel Center')

    # Wheel profile projection στο 3D contact plane
    wheel_2d = results["wheel_2d"]
    plane = results["contact_plane"]
    wheel_3d_points = []
    for point_2d in wheel_2d:
        point_3d = plane['point'] + point_2d[0] * plane['x_axis'] + point_2d[1] * plane['y_axis']
        wheel_3d_points.append(point_3d)
    wheel_3d_points = np.array(wheel_3d_points)
    ax1.plot(wheel_3d_points[:, 0], wheel_3d_points[:, 1], wheel_3d_points[:, 2], 
             'r-', linewidth=2, label='Wheel Profile (3D)')
    ax1.scatter(wheel_3d_points[:, 0], wheel_3d_points[:, 1], wheel_3d_points[:, 2], 
                c='red', s=30, alpha=0.6)

    # Σχεδίαση επιπέδου επαφής
    plane_size = 150
    xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                         np.linspace(-plane_size, plane_size, 10))
    
    # Υπολογισμός των 3D σημείων του επιπέδου
    plane_points = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point_2d = np.array([xx[i, j], yy[i, j]])
            point_3d = plane['point'] + point_2d[0] * plane['x_axis'] + point_2d[1] * plane['y_axis']
            plane_points.append(point_3d)
    
    plane_points = np.array(plane_points)
    xx_3d = plane_points[:, 0].reshape(xx.shape)
    yy_3d = plane_points[:, 1].reshape(xx.shape)
    zz_3d = plane_points[:, 2].reshape(xx.shape)
    
    ax1.plot_surface(xx_3d, yy_3d, zz_3d,
                     alpha=0.2, color='cyan', label='Contact Plane')

    # Σχεδίαση αξόνων επιπέδου
    axis_length = 50
    ax1.quiver(plane['point'][0], plane['point'][1], plane['point'][2],
               plane['x_axis'][0] * axis_length, plane['x_axis'][1] * axis_length, plane['x_axis'][2] * axis_length,
               color='red', linewidth=2, label='Plane X-axis')
    ax1.quiver(plane['point'][0], plane['point'][1], plane['point'][2],
               plane['y_axis'][0] * axis_length, plane['y_axis'][1] * axis_length, plane['y_axis'][2] * axis_length,
               color='green', linewidth=2, label='Plane Y-axis')
    ax1.quiver(plane['point'][0], plane['point'][1], plane['point'][2],
               plane['normal'][0] * axis_length, plane['normal'][1] * axis_length, plane['normal'][2] * axis_length,
               color='blue', linewidth=2, label='Plane Normal')

    # Σχεδίαση δυνάμεων επαφής
    contact_forces = results["contact_forces"]
    if contact_forces:
        max_force = max([np.linalg.norm(f['F_elastic'] + f['F_damp'] + f['F_friction']) for f in contact_forces])
        force_scale_factor = force_scale / max_force if max_force > 0 else force_scale
        
        for i, force_info in enumerate(contact_forces):
            point = force_info['pt']
            F_total_pt = force_info['F_elastic'] + force_info['F_damp'] + force_info['F_friction']
            magnitude = np.linalg.norm(F_total_pt)
            
            # Κλίμακα βάσει μεγέθους δύναμης
            scaled_force = F_total_pt * force_scale_factor
            
            # Χρώμα βάσει μεγέθους δύναμης
            color_intensity = magnitude / max_force if max_force > 0 else 0
            color = (color_intensity, 0, 1 - color_intensity)  # Από μπλε (μικρή) έως κόκκινη (μεγάλη)
            
            # Σχεδίαση διανύσματος δύναμης
            ax1.quiver(point[0], point[1], point[2],
                      scaled_force[0], scaled_force[1], scaled_force[2],
                      color=color, linewidth=2, arrow_length_ratio=0.3,
                      label='Contact Force' if i == 0 else "")
    
    # Σχεδίαση συνολικής δύναμης στο κέντρο του τροχού
    total_force = results["F_total"]
    if np.linalg.norm(total_force) > 0:
        force_scale_factor_total = force_scale / np.linalg.norm(total_force) if np.linalg.norm(total_force) > 0 else force_scale
        scaled_total_force = total_force * force_scale_factor_total
        ax1.quiver(wheel_center[0], wheel_center[1], wheel_center[2],
                  scaled_total_force[0], scaled_total_force[1], scaled_total_force[2],
                  color='black', linewidth=3, arrow_length_ratio=0.3,
                  label='Total Force')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_xlim(-1000, -500)
    ax1.set_ylim(0, 500)
    ax1.set_zlim(10000-250, 10000+250)

    ax1.set_title('3D Visualization of Wheel-Rail Contact with Forces')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

    # 2D Plot (επίπεδο επαφής)
    ax2 = fig.add_subplot(122)

    # Σχεδίαση rail profile στο 2D επίπεδο
    if 'rail_intersection_2d' in results and len(results['rail_intersection_2d']) > 0:
        rail_2d = results['rail_intersection_2d']
        ax2.scatter(rail_2d[:, 0], rail_2d[:, 1], c='blue', s=50, label='Rail Intersection Points')
        
        # Σχεδίαση του polygon αν υπάρχει
        if 'rail_poly' in results and len(results['rail_poly']) > 0:
            ax2.plot(results['rail_poly'][:, 0], results['rail_poly'][:, 1], 
                    'b-', linewidth=2, label='Rail Intersection Polygon')

    # Σχεδίαση wheel profile στο 2D επίπεδο
    wheel_2d = results["wheel_2d"]
    ax2.plot(wheel_2d[:, 0], wheel_2d[:, 1], 'r-', linewidth=2, label='Wheel Profile')

    # Σχεδίαση περιοχών επαφής
    contact_segments = results["contact_segments"]
    contact_label_added = False
    for poly in contact_segments:
        if len(poly) > 1:
            if not contact_label_added:
                ax2.plot(poly[:, 0], poly[:, 1], 'g-', linewidth=3, label='Contact Area')
                contact_label_added = True
            else:
                ax2.plot(poly[:, 0], poly[:, 1], 'g-', linewidth=3)

    ax2.scatter(rail_2d[:, 0], rail_2d[:, 1], c='blue', s=20, alpha=0.6, label='Rail Points')
    ax2.scatter(wheel_2d[:, 0], wheel_2d[:, 1], c='red', s=20, alpha=0.6, label='Wheel Points')

    axis_length_2d = 20
    ax2.quiver(0, 0, axis_length_2d, 0, color='red', linewidth=2, label='X-axis', scale=1, scale_units='xy')
    ax2.quiver(0, 0, 0, axis_length_2d, color='green', linewidth=2, label='Y-axis', scale=1, scale_units='xy')

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Contact Plane Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()

    # Εκτύπωση πληροφοριών
    print(f"\n--- Results ---")
    print(f"Wheel center: {wheel_center}")
    print(f"Plane origin: {plane['point']}")
    print(f"Total Force: {results['F_total']} N")
    print(f"Total Moment: {results['M_total']} N·mm")
    print(f"Number of contact segments: {len(contact_segments)}")
    print(f"Number of contact forces: {len(contact_forces)}")

# MAIN/PARADIGMA ΧΡΗΣΗΣ (ίδιο με untitled50.py)
if __name__ == "__main__":
    # ΧΡΗΣΙΜΟΠΟΙΗΣΕ ΙΔΙΟ wheel_center, V, omega, euler_angles, wheel_type ΟΠΩΣ untitled50.py για να δουλεύει ίδια!
    wheel_center = [-746, 800, 10000]
    V = [0, 0, 1000]
    omega = [0, 10, 0]
    euler_angles = [0/57.3, 0.0/57.3, 0]
    results = GFOSUB_enhanced_analytical(
        wheel_center, V, omega, euler_angles,
        wheel_type="right", return_plot_data=True
    )
    plot_3d_results_with_forces(results, 0.1)
