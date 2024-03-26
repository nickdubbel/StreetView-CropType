# Import google_streetview for the api module
import google_streetview.api

from KEYS_GoogleMaps import KEY, key

# Define parameters for street view api
params = [{
	'size': '600x300', # max 640x640 pixels
	'location': '46.414382,10.013988',
	'heading': '151.78',
	'pitch': '-0.76',
	'key': KEY
}]

# Create a results object
results = google_streetview.api.results(params)

# Download images to directory 'downloads'
results.download_links('downloads')