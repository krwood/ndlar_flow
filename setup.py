import setuptools

with open('README.rst', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('VERSION', 'r') as fh:
    version = fh.read().strip()

setuptools.setup(name='ndlar_flow',
                 version=version,
                 description='An h5flow-based analysis framework for (multi-module) ndlar simulation.',
                 long_description=long_description,
                 long_description_content_type='text/x-rst',
                 author='Peter Madigan, Stephen Greenberg',
                 author_email='pmadigan@berkeley.edu, kwood@lbl.gov',
                 package_dir={'ndlar_flow': './h5flow_modules/'},
                 packages=[p.replace('h5flow_modules', 'ndlar_flow') for p in setuptools.find_packages(where='.')],
                 python_requires='>=3.7',
                 install_requires=[
                     'h5py>=2.10',
                     'pytest',
                     'scipy',
                     'scikit-image',
                     'scikit-learn',
                     'h5flow>=0.1.0'
                 ]
                 )
