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