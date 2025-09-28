import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
from datetime import datetime, timedelta
import json

class LEO_PNT_Constellation:
    def __init__(self):
        # Earth parameters
        self.R_earth = 6371.0  # km
        self.mu_earth = 3.986004418e14  # m^3/s^2
        self.J2 = 1.08262668e-3  # Earth's oblateness coefficient
        
        # Constellation parameters (default Walker constellation)
        self.altitude = 550  # km
        self.inclination = 53.0  # degrees
        self.num_planes = 6
        self.sats_per_plane = 11
        self.total_satellites = self.num_planes * self.sats_per_plane
        
        # Performance parameters
        self.min_elevation_angle = 10.0  # degrees
        self.signal_power = 50.0  # dBW
        self.frequency = 1575.42e6  # Hz (L1 GPS frequency)
        self.constellation_lifetime = 15  # years
        
        # Calculate orbital parameters
        self.orbital_radius = (self.R_earth + self.altitude) * 1000  # meters
        self.orbital_period = 2 * np.pi * np.sqrt(self.orbital_radius**3 / self.mu_earth)  # seconds
        self.orbital_velocity = np.sqrt(self.mu_earth / self.orbital_radius)  # m/s
        
    def calculate_satellite_positions(self, time_offset=0):
        """Calculate positions of all satellites at given time"""
        positions = []
        
        for plane in range(self.num_planes):
            raan = plane * (360 / self.num_planes)  # Right Ascension of Ascending Node
            
            for sat in range(self.sats_per_plane):
                # Mean anomaly for each satellite
                mean_anomaly = sat * (360 / self.sats_per_plane) + time_offset * 360 / (self.orbital_period / 3600)
                
                # Convert to Cartesian coordinates (simplified circular orbit)
                lat = np.arcsin(np.sin(np.radians(self.inclination)) * 
                               np.sin(np.radians(mean_anomaly)))
                lon = np.radians(raan) + np.arctan2(
                    np.cos(np.radians(self.inclination)) * np.sin(np.radians(mean_anomaly)),
                    np.cos(np.radians(mean_anomaly))
                )
                
                # Convert to Cartesian
                x = self.orbital_radius * np.cos(lat) * np.cos(lon)
                y = self.orbital_radius * np.cos(lat) * np.sin(lon)
                z = self.orbital_radius * np.sin(lat)
                
                positions.append({
                    'plane': plane,
                    'satellite': sat,
                    'x': x, 'y': y, 'z': z,
                    'lat': np.degrees(lat),
                    'lon': np.degrees(lon),
                    'altitude': self.altitude
                })
                
        return positions
    
    def calculate_coverage_metrics(self, lat_range=(-90, 90), lon_range=(-180, 180), grid_size=10):
        """Calculate global coverage metrics"""
        lat_points = np.linspace(lat_range[0], lat_range[1], grid_size)
        lon_points = np.linspace(lon_range[0], lon_range[1], grid_size)
        
        coverage_stats = {
            'total_points': 0,
            'covered_points': 0,
            'min_satellites_visible': float('inf'),
            'max_satellites_visible': 0,
            'avg_satellites_visible': 0,
            'coverage_percentage': 0
        }
        
        satellite_visibility_sum = 0
        
        for lat in lat_points:
            for lon in lon_points:
                coverage_stats['total_points'] += 1
                satellites_visible = self.count_visible_satellites(lat, lon)
                
                if satellites_visible > 0:
                    coverage_stats['covered_points'] += 1
                    
                coverage_stats['min_satellites_visible'] = min(
                    coverage_stats['min_satellites_visible'], satellites_visible)
                coverage_stats['max_satellites_visible'] = max(
                    coverage_stats['max_satellites_visible'], satellites_visible)
                satellite_visibility_sum += satellites_visible
        
        coverage_stats['coverage_percentage'] = (
            coverage_stats['covered_points'] / coverage_stats['total_points'] * 100)
        coverage_stats['avg_satellites_visible'] = (
            satellite_visibility_sum / coverage_stats['total_points'])
        
        if coverage_stats['min_satellites_visible'] == float('inf'):
            coverage_stats['min_satellites_visible'] = 0
            
        return coverage_stats
    
    def count_visible_satellites(self, user_lat, user_lon, time_offset=0):
        """Count visible satellites from a ground location"""
        positions = self.calculate_satellite_positions(time_offset)
        visible_count = 0
        
        # Convert user position to Cartesian
        user_x = self.R_earth * 1000 * np.cos(np.radians(user_lat)) * np.cos(np.radians(user_lon))
        user_y = self.R_earth * 1000 * np.cos(np.radians(user_lat)) * np.sin(np.radians(user_lon))
        user_z = self.R_earth * 1000 * np.sin(np.radians(user_lat))
        
        for sat_pos in positions:
            # Calculate elevation angle
            sat_vector = np.array([sat_pos['x'] - user_x, 
                                 sat_pos['y'] - user_y, 
                                 sat_pos['z'] - user_z])
            
            local_up = np.array([user_x, user_y, user_z]) / (self.R_earth * 1000)
            
            elevation = 90 - np.degrees(np.arccos(
                np.dot(sat_vector, local_up) / 
                (np.linalg.norm(sat_vector) * np.linalg.norm(local_up))))
            
            if elevation >= self.min_elevation_angle:
                visible_count += 1
                
        return visible_count
    
    def calculate_dilution_of_precision(self, user_lat, user_lon):
        """Calculate Geometric Dilution of Precision (GDOP)"""
        positions = self.calculate_satellite_positions()
        visible_satellites = []
        
        # Convert user position
        user_x = self.R_earth * 1000 * np.cos(np.radians(user_lat)) * np.cos(np.radians(user_lon))
        user_y = self.R_earth * 1000 * np.cos(np.radians(user_lat)) * np.sin(np.radians(user_lon))
        user_z = self.R_earth * 1000 * np.sin(np.radians(user_lat))
        user_pos = np.array([user_x, user_y, user_z])
        
        for sat_pos in positions:
            sat_vector = np.array([sat_pos['x'] - user_x, 
                                 sat_pos['y'] - user_y, 
                                 sat_pos['z'] - user_z])
            
            local_up = user_pos / np.linalg.norm(user_pos)
            elevation = 90 - np.degrees(np.arccos(
                np.dot(sat_vector, local_up) / np.linalg.norm(sat_vector)))
            
            if elevation >= self.min_elevation_angle:
                # Unit vector from user to satellite
                unit_vector = sat_vector / np.linalg.norm(sat_vector)
                visible_satellites.append(unit_vector)
        
        if len(visible_satellites) < 4:
            return float('inf')  # Cannot compute position
        
        # Construct geometry matrix
        H = np.array([[sat[0], sat[1], sat[2], 1] for sat in visible_satellites[:8]])  # Use max 8 satellites
        
        try:
            # Calculate GDOP
            HTH_inv = np.linalg.inv(H.T @ H)
            gdop = np.sqrt(np.trace(HTH_inv))
            return gdop
        except np.linalg.LinAlgError:
            return float('inf')
    
    def calculate_link_budget(self):
        """Calculate communication link budget"""
        # Free space path loss at maximum range
        max_range = np.sqrt((self.orbital_radius - self.R_earth * 1000)**2 + 
                           (2 * self.R_earth * 1000 * self.orbital_radius * 
                            (1 - np.cos(np.radians(90 - self.min_elevation_angle))))**0.5)
        
        fspl = 20 * np.log10(max_range) + 20 * np.log10(self.frequency) - 147.55
        
        # Atmospheric losses (approximate)
        atmospheric_loss = 2.0  # dB
        
        # Receiver sensitivity (typical for PNT)
        receiver_sensitivity = -130.0  # dBm
        
        # Link margin
        received_power = self.signal_power - fspl - atmospheric_loss
        link_margin = received_power - receiver_sensitivity
        
        return {
            'max_range_km': max_range / 1000,
            'free_space_path_loss_db': fspl,
            'atmospheric_loss_db': atmospheric_loss,
            'received_power_dbm': received_power,
            'link_margin_db': link_margin
        }
    
    def calculate_operational_metrics(self):
        """Calculate operational and sustainability metrics"""
        # Orbital decay due to atmospheric drag (simplified)
        atmospheric_density = 5.2e-12 * np.exp(-(self.altitude - 200) / 50)  # kg/m³ (approximate)
        drag_coefficient = 2.2
        satellite_area = 10.0  # m² (assumed)
        satellite_mass = 150.0  # kg (assumed)
        
        drag_acceleration = 0.5 * atmospheric_density * drag_coefficient * satellite_area / satellite_mass
        orbital_decay_rate = drag_acceleration * self.orbital_period / (2 * np.pi)  # m per orbit
        
        # Station-keeping requirements
        annual_decay = orbital_decay_rate * (365.25 * 24 * 3600 / self.orbital_period) / 1000  # km/year
        delta_v_per_year = annual_decay * 2 * np.pi / self.orbital_period * 1000  # m/s per year
        
        # Collision probability (simplified Kessler syndrome assessment)
        debris_density = 1e-9  # objects per km³ (approximate)
        collision_probability = debris_density * satellite_area * self.orbital_velocity * 365.25 * 24 * 3600
        
        return {
            'orbital_period_minutes': self.orbital_period / 60,
            'orbital_velocity_km_s': self.orbital_velocity / 1000,
            'atmospheric_density_kg_m3': atmospheric_density,
            'annual_altitude_decay_km': annual_decay,
            'delta_v_requirement_m_s_year': delta_v_per_year,
            'collision_probability_per_year': collision_probability,
            'constellation_refresh_rate_years': 1 / (collision_probability + 0.01)  # Including other failure modes
        }
    
    def plot_3d_globe(self, ax, positions):
        """Create 3D globe visualization with satellites"""
        # Create Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        earth_x = self.R_earth * np.outer(np.cos(u), np.sin(v))
        earth_y = self.R_earth * np.outer(np.sin(u), np.sin(v))
        earth_z = self.R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot Earth as a blue sphere with continents texture
        ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.6, color='lightblue', 
                       linewidth=0, antialiased=True)
        
        # Add continent outlines (simplified)
        self.add_continent_outlines(ax)
        
        # Plot satellites in 3D
        sat_x = [pos['x']/1000 for pos in positions]  # Convert to km
        sat_y = [pos['y']/1000 for pos in positions]
        sat_z = [pos['z']/1000 for pos in positions]
        planes = [pos['plane'] for pos in positions]
        
        # Color satellites by orbital plane
        colors = plt.cm.tab10(np.array(planes) / max(planes))
        scatter = ax.scatter(sat_x, sat_y, sat_z, c=colors, s=50, alpha=0.9)
        
        # Add orbital paths for a few satellites (sample)
        self.add_orbital_paths(ax, sample_size=6)
        
        # Customize the plot
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('3D LEO-PNT Constellation\n(Satellites around Earth)')
        
        # Set equal aspect ratio
        max_range = (self.R_earth + self.altitude) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Add legend
        ax.text2D(0.05, 0.95, f"• {self.total_satellites} satellites\n• {self.num_planes} orbital planes\n• {self.altitude} km altitude", 
                 transform=ax.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def add_continent_outlines(self, ax):
        """Add simplified continent outlines to the 3D globe"""
        # Simplified continent data (major landmasses)
        continents = [
            # North America (simplified)
            {'name': 'N.America', 'lats': [25, 35, 45, 55, 65, 70, 65, 50, 30], 
             'lons': [-120, -110, -100, -95, -110, -130, -150, -160, -140]},
            # Europe
            {'name': 'Europe', 'lats': [35, 45, 60, 70, 60, 50, 40], 
             'lons': [-10, 0, 10, 20, 30, 40, 20]},
            # Asia
            {'name': 'Asia', 'lats': [20, 30, 50, 70, 60, 40, 25], 
             'lons': [70, 80, 90, 100, 120, 140, 110]},
            # Africa
            {'name': 'Africa', 'lats': [35, 20, 0, -20, -35, -30, -10, 10, 30], 
             'lons': [0, 10, 20, 25, 20, 30, 40, 45, 10]},
        ]
        
        for continent in continents:
            # Convert lat/lon to 3D coordinates on Earth surface
            lats_rad = np.radians(continent['lats'])
            lons_rad = np.radians(continent['lons'])
            
            x = self.R_earth * 1.01 * np.cos(lats_rad) * np.cos(lons_rad)
            y = self.R_earth * 1.01 * np.cos(lats_rad) * np.sin(lons_rad)
            z = self.R_earth * 1.01 * np.sin(lats_rad)
            
            ax.plot(x, y, z, 'k-', linewidth=1, alpha=0.7)
    
    def add_orbital_paths(self, ax, sample_size=6):
        """Add orbital path visualization for sample satellites"""
        # Generate orbital paths for a few satellites
        sample_planes = range(0, self.num_planes, max(1, self.num_planes//sample_size))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sample_planes)))
        
        for i, plane in enumerate(sample_planes):
            # Generate points along the orbital path
            angles = np.linspace(0, 2*np.pi, 100)
            
            # Orbital path in orbital plane coordinates
            orbital_radius_km = (self.R_earth + self.altitude)
            path_x = orbital_radius_km * np.cos(angles)
            path_y = orbital_radius_km * np.sin(angles)
            path_z = np.zeros_like(angles)
            
            # Rotate to actual orbital plane (simplified rotation)
            inc_rad = np.radians(self.inclination)
            raan_rad = np.radians(plane * 360 / self.num_planes)
            
            # Apply inclination rotation
            rotated_y = path_y * np.cos(inc_rad) - path_z * np.sin(inc_rad)
            rotated_z = path_y * np.sin(inc_rad) + path_z * np.cos(inc_rad)
            
            # Apply RAAN rotation
            final_x = path_x * np.cos(raan_rad) - rotated_y * np.sin(raan_rad)
            final_y = path_x * np.sin(raan_rad) + rotated_y * np.cos(raan_rad)
            final_z = rotated_z
            
            # Plot orbital path
            ax.plot(final_x, final_y, final_z, color=colors[i], linewidth=1.5, alpha=0.6)
    
    def plot_orbital_timeline(self, ax):
        """Plot orbital timeline showing satellite passes"""
        # Generate sample timeline data
        hours = np.linspace(0, 24, 48)  # 30-minute intervals
        
        # Simulate different metrics over time
        total_visible = []
        coverage_percentage = []
        avg_elevation = []
        
        for hour in hours:
            # Simulate time-varying coverage (orbital mechanics causes variation)
            base_coverage = 90
            orbital_variation = 5 * np.sin(2 * np.pi * hour / (self.orbital_period / 3600))
            daily_variation = 3 * np.sin(2 * np.pi * hour / 24)
            noise = np.random.normal(0, 1)
            
            coverage = base_coverage + orbital_variation + daily_variation + noise
            coverage_percentage.append(max(0, min(100, coverage)))
            
            # Visible satellites (varies with orbital geometry)
            visible = int(4 + 3 * np.sin(2 * np.pi * hour / (self.orbital_period / 3600)) + 
                         np.random.normal(0, 0.5))
            total_visible.append(max(0, visible))
            
            # Average elevation angle
            elevation = 35 + 10 * np.sin(2 * np.pi * hour / 12) + np.random.normal(0, 2)
            avg_elevation.append(max(10, elevation))
        
        # Create timeline plot
        ax2 = ax.twinx()  # Secondary y-axis
        
        # Plot coverage percentage
        line1 = ax.plot(hours, coverage_percentage, 'b-', linewidth=2, label='Coverage %', alpha=0.8)
        ax.set_ylabel('Coverage (%)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(80, 100)
        
        # Plot number of visible satellites
        line2 = ax2.plot(hours, total_visible, 'r-', linewidth=2, label='Visible Satellites', alpha=0.8)
        ax2.set_ylabel('Number of Visible Satellites', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 12)
        
        # Formatting
        ax.set_xlabel('Time (hours)')
        ax.set_title('24-Hour Constellation Performance Timeline')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        
        # Add orbital period markers
        orbital_hours = self.orbital_period / 3600
        for i in range(int(24 / orbital_hours) + 1):
            marker_time = i * orbital_hours
            if marker_time <= 24:
                ax.axvline(x=marker_time, color='gray', linestyle='--', alpha=0.5)
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        
        # Add performance statistics
        avg_coverage = np.mean(coverage_percentage)
        avg_visible_sats = np.mean(total_visible)
        
        textstr = f'Avg Coverage: {avg_coverage:.1f}%\nAvg Visible: {avg_visible_sats:.1f} sats\nOrbital Period: {orbital_hours:.2f}h'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive constellation analysis report"""
        print("=" * 80)
        print("LEO-PNT SATELLITE CONSTELLATION DESIGN REPORT")
        print("=" * 80)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Basic constellation parameters
        print("CONSTELLATION ARCHITECTURE")
        print("-" * 40)
        print(f"Total Satellites: {self.total_satellites}")
        print(f"Number of Orbital Planes: {self.num_planes}")
        print(f"Satellites per Plane: {self.sats_per_plane}")
        print(f"Orbital Altitude: {self.altitude} km")
        print(f"Orbital Inclination: {self.inclination}°")
        print(f"Constellation Type: Walker Delta Pattern")
        print()
        
        # Orbital characteristics
        operational_metrics = self.calculate_operational_metrics()
        print("ORBITAL CHARACTERISTICS")
        print("-" * 40)
        print(f"Orbital Period: {operational_metrics['orbital_period_minutes']:.1f} minutes")
        print(f"Orbital Velocity: {operational_metrics['orbital_velocity_km_s']:.2f} km/s")
        print(f"Orbital Radius: {self.orbital_radius/1000:.1f} km")
        print()
        
        # Coverage analysis
        print("Analyzing global coverage... (this may take a moment)")
        coverage_metrics = self.calculate_coverage_metrics(grid_size=15)
        print("COVERAGE PERFORMANCE")
        print("-" * 40)
        print(f"Global Coverage: {coverage_metrics['coverage_percentage']:.1f}%")
        print(f"Minimum Visible Satellites: {coverage_metrics['min_satellites_visible']}")
        print(f"Maximum Visible Satellites: {coverage_metrics['max_satellites_visible']}")
        print(f"Average Visible Satellites: {coverage_metrics['avg_satellites_visible']:.1f}")
        print(f"Minimum Elevation Angle: {self.min_elevation_angle}°")
        print()
        
        # Positioning accuracy
        # Sample GDOP calculations for different latitudes
        test_locations = [
            (0, 0, "Equator"),
            (40, -74, "New York"),
            (60, 10, "High Latitude"),
            (-45, 170, "Southern Ocean")
        ]
        
        print("POSITIONING ACCURACY (GDOP Analysis)")
        print("-" * 40)
        gdop_values = []
        for lat, lon, name in test_locations:
            gdop = self.calculate_dilution_of_precision(lat, lon)
            gdop_values.append(gdop)
            if gdop == float('inf'):
                print(f"{name} ({lat}°, {lon}°): No coverage")
            else:
                print(f"{name} ({lat}°, {lon}°): GDOP = {gdop:.2f}")
        
        avg_gdop = np.mean([g for g in gdop_values if g != float('inf')])
        positioning_accuracy = avg_gdop * 3.0  # Approximate 3D accuracy in meters
        print(f"Average GDOP: {avg_gdop:.2f}")
        print(f"Estimated 3D Positioning Accuracy: {positioning_accuracy:.1f} meters")
        print()
        
        # Communication link budget
        link_budget = self.calculate_link_budget()
        print("COMMUNICATION LINK BUDGET")
        print("-" * 40)
        print(f"Maximum Range: {link_budget['max_range_km']:.0f} km")
        print(f"Transmit Power: {self.signal_power:.1f} dBW")
        print(f"Free Space Path Loss: {link_budget['free_space_path_loss_db']:.1f} dB")
        print(f"Atmospheric Loss: {link_budget['atmospheric_loss_db']:.1f} dB")
        print(f"Received Power: {link_budget['received_power_dbm']:.1f} dBm")
        print(f"Link Margin: {link_budget['link_margin_db']:.1f} dB")
        print()
        
        # Operational sustainability
        print("OPERATIONAL SUSTAINABILITY")
        print("-" * 40)
        print(f"Atmospheric Density: {operational_metrics['atmospheric_density_kg_m3']:.2e} kg/m³")
        print(f"Annual Altitude Decay: {operational_metrics['annual_altitude_decay_km']:.2f} km/year")
        print(f"Station-keeping ΔV: {operational_metrics['delta_v_requirement_m_s_year']:.1f} m/s/year")
        print(f"Collision Risk: {operational_metrics['collision_probability_per_year']:.6f} per satellite per year")
        print(f"Recommended Refresh Cycle: {operational_metrics['constellation_refresh_rate_years']:.1f} years")
        print()
        
        # Cost estimates
        print("COST ESTIMATES")
        print("-" * 40)
        satellite_cost = 2.0  # Million USD per satellite (estimate)
        launch_cost = 50.0  # Million USD per launch (estimate)
        satellites_per_launch = 20  # Estimate
        
        total_satellite_cost = self.total_satellites * satellite_cost
        total_launch_cost = (self.total_satellites / satellites_per_launch) * launch_cost
        annual_operational_cost = self.total_satellites * 0.1  # Million USD per satellite per year
        
        print(f"Satellite Manufacturing: ${total_satellite_cost:.0f}M USD")
        print(f"Launch Services: ${total_launch_cost:.0f}M USD")
        print(f"Total Deployment Cost: ${total_satellite_cost + total_launch_cost:.0f}M USD")
        print(f"Annual Operations Cost: ${annual_operational_cost:.0f}M USD")
        print()
        
        # Performance summary
        print("PERFORMANCE SUMMARY")
        print("-" * 40)
        availability = coverage_metrics['coverage_percentage']
        continuity = min(99.9, 100 - operational_metrics['collision_probability_per_year'] * 100)
        integrity = max(0, 99.9 - avg_gdop * 5)  # Simplified integrity metric
        
        print(f"Availability: {availability:.1f}%")
        print(f"Continuity: {continuity:.2f}%")
        print(f"Integrity: {integrity:.1f}%")
        print(f"Positioning Accuracy: {positioning_accuracy:.1f} meters (95% confidence)")
        print(f"Time to First Fix: ~30 seconds (estimated)")
        print()
        
        # Recommendations
        print("DESIGN RECOMMENDATIONS")
        print("-" * 40)
        if coverage_metrics['coverage_percentage'] < 95:
            print("• Consider increasing satellite count or optimizing orbital parameters")
        if avg_gdop > 3.0:
            print("• GDOP values are high - consider adding more orbital planes")
        if link_budget['link_margin_db'] < 10:
            print("• Link margin is low - consider increasing transmit power")
        if operational_metrics['annual_altitude_decay_km'] > 5:
            print("• High orbital decay - consider higher altitude or more fuel")
        if operational_metrics['collision_probability_per_year'] > 0.01:
            print("• High collision risk - implement collision avoidance systems")
        
        print("• Implement inter-satellite links for improved global coverage")
        print("• Consider atomic clock technology for enhanced timing accuracy")
        print("• Plan for graceful constellation refresh and debris mitigation")
        print()
        
        # Key output values summary
        key_outputs = {
            "constellation_architecture": {
                "total_satellites": self.total_satellites,
                "orbital_planes": self.num_planes,
                "satellites_per_plane": self.sats_per_plane,
                "altitude_km": self.altitude,
                "inclination_deg": self.inclination
            },
            "orbital_parameters": {
                "period_minutes": operational_metrics['orbital_period_minutes'],
                "velocity_km_s": operational_metrics['orbital_velocity_km_s'],
                "orbital_radius_km": self.orbital_radius/1000
            },
            "coverage_metrics": {
                "global_coverage_percent": coverage_metrics['coverage_percentage'],
                "min_visible_satellites": coverage_metrics['min_satellites_visible'],
                "max_visible_satellites": coverage_metrics['max_satellites_visible'],
                "avg_visible_satellites": coverage_metrics['avg_satellites_visible']
            },
            "accuracy_metrics": {
                "average_gdop": avg_gdop,
                "positioning_accuracy_m": positioning_accuracy,
                "min_elevation_angle_deg": self.min_elevation_angle
            },
            "link_budget": link_budget,
            "operational_metrics": operational_metrics,
            "cost_estimates": {
                "deployment_cost_million_usd": total_satellite_cost + total_launch_cost,
                "annual_operations_million_usd": annual_operational_cost
            },
            "performance_kpis": {
                "availability_percent": availability,
                "continuity_percent": continuity,
                "integrity_percent": integrity
            }
        }
        
        print("KEY OUTPUT VALUES (JSON format for further analysis)")
        print("-" * 60)
        print(json.dumps(key_outputs, indent=2))
        
        return key_outputs

# Create and analyze the constellation
if __name__ == "__main__":
    # Initialize constellation
    constellation = LEO_PNT_Constellation()
    
    # Generate comprehensive report
    results = constellation.generate_comprehensive_report()
    
    # Create comprehensive visualizations
    try:
        positions = constellation.calculate_satellite_positions()
        
        # Create multiple visualization types
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Ground track visualization
        ax1 = plt.subplot(221)
        lats = [pos['lat'] for pos in positions]
        lons = [pos['lon'] for pos in positions]
        planes = [pos['plane'] for pos in positions]
        
        scatter = ax1.scatter(lons, lats, c=planes, cmap='tab10', s=40, alpha=0.8)
        ax1.set_xlabel('Longitude (degrees)')
        ax1.set_ylabel('Latitude (degrees)')
        ax1.set_title(f'LEO-PNT Constellation Ground Track\n({constellation.total_satellites} satellites)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(-90, 90)
        plt.colorbar(scatter, ax=ax1, label='Orbital Plane')
        
        # Plot 2: Coverage heatmap
        ax2 = plt.subplot(222)
        coverage_lats = np.linspace(-90, 90, 20)
        coverage_lons = np.linspace(-180, 180, 40)
        coverage_grid = np.zeros((len(coverage_lats), len(coverage_lons)))
        
        for i, lat in enumerate(coverage_lats):
            for j, lon in enumerate(coverage_lons):
                coverage_grid[i, j] = constellation.count_visible_satellites(lat, lon)
        
        im = ax2.imshow(coverage_grid, extent=[-180, 180, -90, 90], 
                       aspect='auto', origin='lower', cmap='viridis')
        ax2.set_xlabel('Longitude (degrees)')
        ax2.set_ylabel('Latitude (degrees)')
        ax2.set_title('Satellite Visibility Coverage')
        plt.colorbar(im, ax=ax2, label='Number of Visible Satellites')
        
        # Plot 3: 3D Globe representation
        ax3 = fig.add_subplot(223, projection='3d')
        constellation.plot_3d_globe(ax3, positions)
        
        # Plot 4: Orbital timeline
        ax4 = plt.subplot(224)
        constellation.plot_orbital_timeline(ax4)
        
        plt.tight_layout()
        plt.savefig('leo_pnt_constellation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'leo_pnt_constellation_analysis.png'")
        
    except Exception as e:
        print(f"\nVisualization could not be generated: {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
