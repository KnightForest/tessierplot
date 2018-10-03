from setuptools import setup

setup(name='tessierplot',
      version='0.1',
      description='Module for plotting/manipulating 2d/3d data',
      url='http://github.com/wakass/tessierplot',
      author='WakA',
      author_email='alaeca@gmail.com',
      license='MIT',
      packages=['tessierplot'],
      install_requires=[
          'matplotlib',
          'pyperclip'
      ],
      zip_safe=False)