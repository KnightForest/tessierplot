from setuptools import setup, find_packages

setup(name='tessierplot',
      version='0.2',
      description='Module for plotting/manipulating 2d/3d data',
      url='http://github.com/wakass/tessierplot',
      author='WakA',
      author_email='alaeca@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'matplotlib',
          'pyperclip',
          'six',
          'pandas',
          'pywin32',
          'quantiphy'
      ],
      zip_safe=False)