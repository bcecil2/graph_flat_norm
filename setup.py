from distutils.core import setup

setup(
    name='graph_flat_norm',
    version='0.0.7',
    author='Curtis Michels & Blake Cecil',
    author_email='iscicodes@proton.me',
    packages=['graph_flat_norm'],
    url='http://pypi.python.org/pypi/graph_flat_norm/',
    description='Computes the Flat Norm on Graphs',
    long_description=open('README.MD').read(),
    install_requires=[
        "matplotlib",
        "numpy",
        "numba",
        "networkx",
        "opencv-python",
        "scipy"
    ],
)