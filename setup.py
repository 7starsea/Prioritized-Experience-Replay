
from skbuild import setup
from setuptools import find_packages

# # python setup.py install --generator "Sublime Text 2 - Unix Makefiles" -- -- -j8
# # python setup.py install  -- -- -j8
package_folder = 'shark'


setup(
    name='shark',
    version='0.0.1',
    description='prioritized experience replay in reinforcement learning project shark',
    author='Aimin Huang',
    author_email='huangepn@gmail.com',
    packages=find_packages(),  # same as name
    cmake_source_dir="util_cpp/",
)

# print(find_packages())
