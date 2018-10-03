#set the asset directory
import pkg_resources, os
resource_package = __name__
resource_path = 'assets'
asset_directory = pkg_resources.resource_filename(resource_package, resource_path)


def get_asset(filename):
	return os.path.join(asset_directory,filename)
