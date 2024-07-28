import folium
from folium.plugins import HeatMap
import pandas as pd
from sklearn.cluster import DBSCAN
import requests

def create_map():
    def get_data():
        get_data_url = "http://localhost:5000/roadcondition/notifications?APIKey=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c&road_section_id=wtl_a1234"
        response = requests.get(get_data_url)
        data = []
        
        # Check if the request was successful
        if response.status_code in [200, 201]:
            data = response.json()  # If the response contains JSON data
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            
        return data['data']

    data_A = get_data()

    # Convert the data to a DataFrame
    df = pd.DataFrame(data_A)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)

    # Use DBSCAN to cluster nearby points
    coords = df[['latitude', 'longitude']].values
    db = DBSCAN(eps=0.015, min_samples=1).fit(coords)
    df['Cluster'] = db.labels_

    # Group by clusters and count frequencies
    clustered_data = df.groupby('Cluster').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'Cluster': 'size'
    }).rename(columns={'Cluster': 'Frequency'}).reset_index(drop=True)

    # Convert the result to the desired format
    result = {
        'latitude': clustered_data['latitude'].tolist(),
        'longitude': clustered_data['longitude'].tolist(),
        'Frequency': clustered_data['Frequency'].tolist()
    }

    result = pd.DataFrame(result)
    # Create map
    map = folium.Map(location=[43.453509, -80.502824], zoom_start=12)

    # Add heatmap
    HeatMap(data=result[['latitude', 'longitude', 'Frequency']].values, radius=15).add_to(map)

    # Save map to HTML file
    map.save('./streamlit/roadsense_frontend/heatmap.html')

    map
