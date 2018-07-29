from setuptools import setup, find_packages

setup(name='mldatakit',
    version='0.2',
    description='Data conversion and download utility for Machine Learning.',
    url='https://github.com/ml-data-kit/ml-data-kit',
    packages=find_packages(),
    license='Apache',
    scripts=['bin/datakit-flowers102-h5', 'bin/datakit-tiny-imagenet-h5'],
    keywords='Machine Learning, Data, Data Conversion, Data Processing',
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python'
    ],
    include_package_data=True)