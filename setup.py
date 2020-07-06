# coding=utf-8
import setuptools

setuptools.setup(name='zwad',
                 packages=['zwad'],
                 install_requires=[
                     'pandas',
                     'click',
                     'pillow',
                     'scikit-learn',
                 ],
                 entry_points={
                     'console_scripts': [
                         'zwadp = zwad.ad:ZtfAnomalyDetector.script'
                     ]
                 })
