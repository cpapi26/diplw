def GFOSUB_enhanced_analytical(wheel_center, V, omega, euler_angles, wheel_type="left", return_plot_data=True):
    """Βελτιωμένη GFOSUB με αναλυτική προσέγγιση - με έλεγχο επαφής"""
    
    # Μετατροπή Euler angles σε quaternion
    rot = R.from_euler('zyx', euler_angles)
    wheel_quat = rot.as_quat()
    
    # Δημιουργία βελτιωμένου συστήματος
    system = WheelRailSystemAnalytical(wheel_center, wheel_quat, V, omega, wheel_type="right")
    
    # Ορισμός twist function
    twist_func = RailTwistFunction(
        s_points=[0, 50000, 120000],
        theta_points=[0, 10/57.30, 0.0],
        degrees=False
    )
    system.set_twist_table(
        list(zip(twist_func.s_points, twist_func.theta_points)),
        degrees=False
    )

    # Αναλυτικό επίπεδο επαφής
    plane = system.get_first_plane_analytical()

    # === Sweep & Intersection ===
    profile_2d = np.column_stack([system.RAIL_POINTS_X, system.RAIL_POINTS_Y])
    s_center = plane['s_position']
    s_window = 500.0
    num_samples = 10
    s_values = np.linspace(s_center-s_window/2, s_center+s_window/2, num_samples)

    # path_func (για sweep): επιστρέφει origin, tangent, normal, binormal
    def path_func(s):
        frame = system.get_twisted_frame(s)
        return frame['origin'], frame['tangent'], frame['normal'], frame['binormal']

    # Sweep all profile points along path
    profile_paths = compute_profile_paths(profile_2d, s_values, path_func, twist_func)

    # Find intersections with contact plane
    rail_intersection_3d, rail_intersection_2d = find_intersections_with_plane(profile_paths, plane)

    # Create rail polygon (Convex Hull)
    rail_poly = create_rail_polygon_from_intersection(rail_intersection_2d)

    # Wheel profile in 2D (local contact plane coords)
    sign_x = 1 if wheel_type == 'left' else -1
    wheel_2d = np.column_stack([sign_x * system.WHEEL_POINTS_X, -system.WHEEL_POINTS_R])

    # Find wheel-rail contact segments
    wheel_contact_polylines = wheel_polygon_contact_segments_multi(wheel_2d, rail_poly)
    print(f"\n--- Contact Analysis ---")
    print(f"Number of contact polylines: {len(wheel_contact_polylines)}")
    
    # --- ΥΠΟΛΟΓΙΣΜΟΣ ΔΥΝΑΜΕΩΝ ---
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
    
    # Εύρεση σημείων επαφής
    for i, poly in enumerate(wheel_contact_polylines):
        print(f"Polyline {i+1}: {len(poly)} points")
        subpolys = split_polyline_by_length(poly, max_length)
        for j, subpoly in enumerate(subpolys):
            if len(subpoly) > 1:
                max_min_dist, contact_vec, wheel_pt, rail_pt = find_max_min_distance_vector_dense(
                    subpoly, rail_poly, samples_per_segment=30
                )
                print(f"  Subpoly {j+1}: max_min_dist = {max_min_dist:.6f}")
                max_min_info.append({
                    "max_min_dist": max_min_dist,
                    "vec": contact_vec,
                    "pt_on_poly": wheel_pt,
                    "pt_on_polygon": rail_pt,
                    "axes": (plane['point'], plane['x_axis'], plane['y_axis'], plane['z_axis'])
                })

    print(f"\n--- Force Calculation ---")
    print(f"Number of contact points found: {len(max_min_info)}")
    
    # Υπολογισμός δυνάμεων για κάθε σημείο επαφής
    for i, info in enumerate(max_min_info):
        penetration_depth = abs(info["max_min_dist"])
        penetration_depth = softplus(penetration_depth, beta=20)
        if penetration_depth > max_penetration_allowed:
            penetration_depth = max_penetration_allowed

        print(f"Contact point {i+1}: penetration = {penetration_depth:.6f}")

        origin, x_axis, y_axis, z_axis = info["axes"]
        Pmax2d = info["pt_on_poly"]
        Qrail2d = info["pt_on_polygon"]
        Pmax = origin + Pmax2d[0] * x_axis + Pmax2d[1] * y_axis
        Qrail = origin + Qrail2d[0] * x_axis + Qrail2d[1] * y_axis

        penetration_dir = Pmax - Qrail
        penetration_dir_norm = np.linalg.norm(penetration_dir)
        if penetration_dir_norm < 1e-12:
            print(f"  Skipping - zero penetration direction")
            continue
        penetration_dir = penetration_dir / penetration_dir_norm

        Fx = np.dot(penetration_dir, x_axis)
        Fy = np.dot(penetration_dir, y_axis)
        Fz = np.dot(penetration_dir, z_axis)
        penetration_dir_mod = Fx * x_axis + Fy * y_axis + Fz * z_axis
        penetration_dir_mod_norm = np.linalg.norm(penetration_dir_mod)
        if penetration_dir_mod_norm < 1e-12:
            print(f"  Skipping - zero modified penetration direction")
            continue
        penetration_dir_mod = penetration_dir_mod / penetration_dir_mod_norm

        # Elastic force
        F_elastic = -K * (penetration_depth ** n) * penetration_dir_mod
        
        # Damping force
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

        # Tangential velocity and friction
        v_tan = v_point - np.dot(v_point, penetration_dir_mod) * penetration_dir_mod
        v_tan_norm = np.linalg.norm(v_tan)
        v_tan_safe = max(v_tan_norm, 1e-8)
        tangent_unit_vector = v_tan / v_tan_safe

        # Linear friction model
        mu = compute_friction_coefficient(v_tan_norm, mu_dynamic, v_threshold)
        F_friction = -mu * F_normal * tangent_unit_vector
        
        # Moments
        r_vec = Pmax - wheel_center_np
        M_elastic = np.cross(r_vec, F_elastic)
        M_damp = np.cross(r_vec, F_damp)
        M_friction = np.cross(r_vec, F_friction)

        F_total += F_elastic + F_damp + F_friction
        M_total += M_elastic + M_damp + M_friction

        # Print individual forces
        print(f"  F_elastic: {F_elastic}")
        print(f"  F_damp: {F_damp}")
        print(f"  F_friction: {F_friction}")
        print(f"  Total at point: {F_elastic + F_damp + F_friction}")

        forces_plot_main.append({
            'pt': Pmax,
            'F_elastic': F_elastic,
            'F_damp': F_damp,
            'F_friction': F_friction,
            'penetration_dir_mod': penetration_dir_mod,
            'penetration_depth': penetration_depth,
            'tangent_unit_vector': tangent_unit_vector
        })

    # Print total forces
    print(f"\n--- Final Results ---")
    print(f"Total Force: {F_total}")
    print(f"Total Moment: {M_total}")
    print(f"Number of contact forces: {len(forces_plot_main)}")

    # Επιστροφή αποτελεσμάτων
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