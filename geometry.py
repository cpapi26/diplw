def segment_plane_intersection(point_a, point_b, plane):
    """
    Βρίσκει την τομή ευθυγράμμου τμήματος με επίπεδο
    """
    # Διάνυσμα κατεύθυνσης τμήματος
    segment_dir = point_b - point_a
    segment_length = np.linalg.norm(segment_dir)
    if segment_length < 1e-12:
        return None
    
    segment_dir = segment_dir / segment_length
    
    # Παράμετρος τομής
    denom = np.dot(segment_dir, plane['normal'])
    if abs(denom) < 1e-12:  # Παράλληλο με το επίπεδο
        return None
    
    t = np.dot(plane['point'] - point_a, plane['normal']) / denom
    
    # Έλεγχος αν η τομή είναι μέσα στο τμήμα
    if 0 <= t <= segment_length:
        return point_a + t * segment_dir
    return None

def intersect_rail_with_plane_analytical(system, plane, s_center, s_window=100.0, num_samples=500):
    """
    Βρίσκει την τομή του αναλυτικού rail profile με το επίπεδο επαφής
    """
    # Ορισμός εύρους αναζήτησης
    s_min = max(s_center - s_window/2, 0)
    s_max = min(s_center + s_window/2, system.path_func['length'])
    
    s_values = np.linspace(s_min, s_max, num_samples)
    intersection_points_3d = []
    intersection_points_2d = []
    
    # Για κάθε θέση s κατά μήκος της διαδρομής
    for i, s in enumerate(s_values):
        # Παίρνουμε το rail profile σε αυτό το s (ΑΝΑΛΥΤΙΚΑ)
        rail_profile_3d, frame = system.get_rail_profile_at_s(s)
        
        # Βρίσκουμε την τομή κάθε ευθύγραμμου τμήματος του profile με το επίπεδο
        for j in range(len(rail_profile_3d) - 1):
            point_a = rail_profile_3d[j]
            point_b = rail_profile_3d[j + 1]
            
            # Έλεγχος τομής ευθυγράμμου τμήματος με επίπεδο
            intersection = segment_plane_intersection(point_a, point_b, plane)
            if intersection is not None:
                intersection_points_3d.append(intersection)
                
                # Προβολή στο 2D επίπεδο
                rel = intersection - plane['point']
                x_local = np.dot(rel, plane['x_axis'])
                y_local = np.dot(rel, plane['y_axis'])
                intersection_points_2d.append([x_local, y_local])
    
    return np.array(intersection_points_3d), np.array(intersection_points_2d)

def create_rail_polygon_from_intersection(points_2d):
    """
    Δημιουργεί ένα κλειστό polygon από τα σημεία τομής
    """
    if len(points_2d) == 0:
        return np.array([])
    
    # Ταξινόμηση των σημείων για να σχηματίσουμε ένα συνεχές polygon
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points_2d)
        ordered_points = points_2d[hull.vertices]
        # Κλείσιμο του polygon
        return np.vstack([ordered_points, ordered_points[0]])
    except:
        # Εφεδρική λύση αν αποτύχει το convex hull
        return np.vstack([points_2d, points_2d[0]])

def segment_intersection(p1, p2, q1, q2):
    """Τομή ευθυγράμμων τμημάτων"""
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
    """Εύρεση τμημάτων επαφής - τροποποιημένη για πολυγραμμές"""
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
    
    # Μετατροπή σε πολυγραμμές όπως στον δεύτερο κώδικα
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
    """Softplus function"""
    return (1.0 / beta) * np.log(1 + np.exp(beta * x))

def project_to_plane_analytical(points, plane):
    """Προβολή 3D σημείων σε 2D επίπεδο"""
    rel = points - plane['point']
    x_local = np.dot(rel, plane['x_axis'])
    y_local = np.dot(rel, plane['y_axis'])
    return np.column_stack([x_local, y_local])

def rail_polygon_n_lines(profile_2d):
    """Κλείσιμο polygon ράγας"""
    return np.vstack([profile_2d, profile_2d[0]])

def split_polyline_by_length(polyline, max_length):
    """Διαχωρισμός πολυγραμμής σε μικρότερα τμήματα"""
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
    """Εύρεση μέγιστης ελάχιστης απόστασης μεταξύ πολυγραμμών"""
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
    """Δειγματοληψία σημείων κατά μήκος πολυγραμμής"""
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
    """Υπολογισμός συντελεστή τριβής"""
    safe_vd = max(v_d, 1e-12)
    clipped_v = max(v_tan_norm, 0.0)
    return mu_dynamic * min(clipped_v / safe_vd, 1.0)

def sweep_profile_point_3d(profile_point_2d, s, path_func, twist_func):
    origin, tangent, normal, binormal = path_func(s)
    twist_angle = twist_func.twist(s)
    rot = R.from_rotvec(tangent * twist_angle)
    normal_tw = rot.apply(normal)
    binormal_tw = rot.apply(binormal)
    # ΔΙΟΡΘΩΣΗ: Θέλουμε πάντα normal προς τα πάνω
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
            # Υπολογισμός τομής ευθυγράμμου τμήματος με επίπεδο
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