from setuptools import setup

setup(name='VarSelection',
      version='0.1',
      description='Package to support variable selection in classifiers',
      packages=['VarSelection'],
      zip_safe=False)

from distutils.core import setup
setup(
  name = 'VarSelection',        
  packages = ['varSelection'],  
  version = '0.1',      
  license='MIT',        
  description = 'Package to support variable selection in classifiers',  
  author = 'Richard Downey',                  
  author_email = 'downey2k@hotmail.com',      
  url = 'https://github.com/ifu97224/Machine_Learning_Engineer_Nano_Degree/tree/master/',   
  download_url = 'https://github.com/ifu97224/Machine_Learning_Engineer_Nano_Degree/archive/0.1.tar.gz',    
  keywords = ['variable selection'],   
  install_requires=[            
          'pandas',
          'numpy',
          'sklearn',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3.6',
  ],
)