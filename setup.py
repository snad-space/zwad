# coding=utf-8
import setuptools

setuptools.setup(name='zwad',
                 license_files = ('LICENSE',),
                 license='MIT',
                 packages=['zwad', 'zwad.ad'],
                 install_requires=[
                     'coniferest==0.0.13',
                     'pandas<2.0',
                     'click',
                     'pillow',
                     'scikit-learn>=0.23',
                     'seaborn',
                     'tqdm',
                     'matplotlib>=3.1,<4.0',
                     'astropy',
                     'requests>=2,<3.0',
                 ],
                 entry_points={
                     'console_scripts': [
                         'zwaad = zwad.aad:execute_from_commandline',
                         'zwadp = zwad.ad:ZtfAnomalyDetector.script',
                         'zwann = zwad.nn:execute_from_commandline',
                         'zwad-zenodo = zwad.download_feature_set:execute_from_commandline',
                     ]
                 })
