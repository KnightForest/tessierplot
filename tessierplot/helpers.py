import os
from importlib.resources import files

# Define your resource path
asset_directory = files(__package__) / 'assets'

def get_asset(filename):
    return str(asset_directory / filename)
