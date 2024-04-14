import requests
import numpy as np
import math
import pandas as pd
import geopandas
from shapely.geometry import LineString, Point, Polygon

# Constants
EARTH_RADIUS = 6371e3  # in meters
DISTANCE_DELTA = 0.0001  # used for interpolating points along the road
PERPENDICULAR_DISTANCE = 3  # distance for calculating field points
OVERPASS_API_URL = "http://overpass-api.de/api/interpreter"
SHAPEFILE_PATH = 'TreehealthDataset/bomen.shp'
totaal_list = []

def compute_bearing(from_point, to_point):
    """Calculate the bearing from one geographic point to another."""
    y = math.sin(math.radians(to_point[1]) - math.radians(from_point[1])) * math.cos(math.radians(to_point[0]))
    x = math.cos(math.radians(from_point[0])) * math.sin(math.radians(to_point[0])) - \
        math.sin(math.radians(from_point[0])) * math.cos(math.radians(to_point[0])) * math.cos(math.radians(to_point[1] )- math.radians(from_point[1]))
    θ = math.atan2(y, x)
    

    bearing = (θ * 180 / math.pi + 360) % 360
    return bearing

def compute_point_on_field(from_point, theta, distance):
    """Calculate a point at a certain distance and bearing from a given point."""
    angular_distance = distance / EARTH_RADIUS
    theta = math.radians(theta)
    lat1 = math.radians(from_point[0])
    lon1 = math.radians(from_point[1])
    lat2 = math.asin(math.sin(lat1) * math.cos(angular_distance) + \
                     math.cos(lat1) * math.sin(angular_distance) * math.cos(theta))
    lon2 = lon1 + math.atan2(math.sin(theta) * math.sin(angular_distance) * math.cos(lat1),
                             math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))

def process_shapefile(shapefile_path):
    """Process each geometry in the shapefile and save results."""
    geo_data = geopandas.read_file(shapefile_path).to_crs("EPSG:4326")
    # print(geo_data.crs)
    # print(geo_data.keys()) 'WOONPLAATS', 'WIJK', 'BUURT', 'GROENGEBIE', 'GEBIEDCODE', 'ELEMENTNUM',
    #    'BEHEEROBJE', 'BEHEERGROE', 'BOOMSORTIM', 'EXTRA_2', 'BOOMHOOGTE',
    #    'AANLEGJAAR', 'STANDPLAAT', 'STRAAT', 'EIGENAAR', 'BEHEERDER',
    #    'INSPECTIED', 'CONDITIE', 'BOOMBEELD', 'GEBREKEN_A', 'CATEGORIE_',
    #    'GEBREKE_1', 'MAATREGELE', 'VERWACHTE_', 'geometry'
    for geo_idx, geometry in enumerate(geo_data.geometry):
        # if geo_data.iloc[geo_idx]['name_1'] == 'Andhra Pradesh':
        #     print("Geometry ", geo_data.iloc[geo_idx])
        print('GEO Index ', geo_idx)
        conditie = geo_data.CONDITIE[geo_idx]
        if geo_idx >= 0:
            process_geometry(geometry, geo_idx,conditie)
    save_to_csv(totaal_list, f"1200+ .csv", "y,x,b,oy,ox,type")

def process_geometry(geometry, geo_idx,conditie):
    """Process a single geometry from the shapefile."""
    tolerance = 10

    # if geometry.is_complex:  # Adjust this condition based on your criteria for complexity
    # print(len(geometry))
    geometry = geometry.simplify(tolerance, preserve_topology=True)
    # print("Post simplify")
    # print(len(geometry))
    ext_coords = []
    if geometry.type == "MultiPolygon":
        print("MultiPolygon")

        road_data_combined = {'elements': []}  # Initialize combined road data
        for subgeom in geometry.geoms:  # Iterate through each subpolygon
            subgeom_simplified = subgeom.simplify(tolerance, preserve_topology=True)
            ext_coords = list(subgeom_simplified.exterior.coords)
            polygon_query = create_overpass_query(ext_coords)
            road_data = fetch_overpass_data(polygon_query)
            # print(road_data[])
            road_data_combined['elements'].extend(road_data['elements'])  # Combine road data
            # print(road_data)

    else:
        # print("NOT a MULTIPOLYGON")
        geometry = geometry.simplify(tolerance, preserve_topology=True)
        ext_coords = list(geometry.coords)
        # print(geo_idx, ext_coords)
        polygon_query = create_overpass_query(ext_coords)
        road_data = fetch_overpass_data(polygon_query)
        # print(road_data)

    process_road_data(road_data, geo_idx,ext_coords,conditie)

def create_overpass_query(ext_coords):
    """Create an Overpass API query from exterior coordinates."""
    coord_str = " ".join(f"{lon}, {lat}" for lat, lon in ext_coords)
    overpass_query = f"""
    [out:json];
    way(around:30,{coord_str});
    out geom;
    """
    return overpass_query

def fetch_overpass_data(query):
    """Fetch data from the Overpass API."""
    response = requests.get(OVERPASS_API_URL, params={'data': query})
    return response.json()

def process_road_data(road_data, geo_idx,ext_coords,conditie):
    """Process road data and save the output to CSV files."""
    data = [0,0,0,0,0]
    totat_points = [(0,0)]
    road_points, field_points, original_points = [], [], []
    for element in road_data['elements']:
        if element['type'] == 'way':
            keywords = ['highway']
            tags = element.get('tags', {})
            if  any(keyword in tags for keyword in keywords):
                try:
                    process_way_element(element, road_points, field_points, original_points)
                    for i in original_points:
                        totat_points.append(i)
                   

                except Exception as e:
                    print(e)

    coords = [ ext_coords[0][1],ext_coords[0][0]]
    distance = np.linalg.norm(np.array(totat_points)-np.array(coords),axis = 1 )
    min_dist = np.argmin(distance)
    data = list(totat_points[min_dist])
    b = compute_bearing(data, coords)
    data.append(b)
    data.append(coords[0])
    data.append(coords[1])
    data.append(conditie)
    totaal_list.append(data)
    
    # save_to_csv(field_points, f"roadPoints/fieldPointsNW4_{geo_idx}.csv", "y,x,b,yr,xr")
    # save_to_csv(original_points, f"roadPoints/osmRoadsNW4_{geo_idx}.csv", "y,x")

def process_way_element(element, road_points, field_points, original_points):
    """Process a single way element from the Overpass data."""

    geo = element['geometry']
    way = [(p['lat'], p['lon']) for p in geo]
    original_points.extend(way)
    line = LineString(way)
    distances = np.arange(0, line.length, DISTANCE_DELTA)
    points = [line.interpolate(distance) for distance in distances]

    if line.boundary.length > 1:
        points.append(line.boundary[1])
    if len(points) < 2:
        # print("Not enough points to form a LineString")
        return
    new_line = LineString(points)
    # process_line_points(new_line, road_points, field_points)

def process_line_points(line, road_points, field_points):
    """Process points along a line and compute adjacent field points."""
    old_x, old_y = None, None

    for j, (x, y) in enumerate(line.coords):
        if j > 3 and j< len(line.coords)-3:

            from_point = (old_x, old_y)
            to_point = (x, y)
            bearing = compute_bearing(from_point, to_point)
            p1 = compute_point_on_field(to_point, (bearing + 90) % 360, PERPENDICULAR_DISTANCE)
            p2 = compute_point_on_field(to_point, (bearing + 270) % 360, PERPENDICULAR_DISTANCE)
            field_points.append((p1[0], p1[1], (bearing + 90) % 360, x, y))
            field_points.append((p2[0], p2[1], (bearing + 270) % 360, x, y))
            road_points.append((x, y, bearing, p1[0], p1[1], p2[0], p2[1]))
        old_x, old_y = x, y

def save_to_csv(data, filename, header):
    """Save data to a CSV file."""
    np.savetxt(filename, data, delimiter=",", fmt=['%f','%f','%f','%f','%f','%s'], header=header, comments='')

# Main execution
if __name__ == "__main__":
    process_shapefile(SHAPEFILE_PATH)