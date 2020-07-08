# coding=utf-8
import setuptools

setuptools.setup(name='zwad',
                 packages=['zwad'],
                 install_requires=[
                     'pandas<1.0',
                     'click',
                     'pillow',
                     'scikit-learn<0.23',
                     'tqdm',
                     'ad_examples @ git+https://github.com/shubhomoydas/ad_examples.git@60afac80eb0ed7d7da1a02b7718a8d6d305179ab#egg=ad_examples-0.0.0'],
                 entry_points={
                     'console_scripts': [
                         'zwaad = zwad.aad:execute_from_commandline',
                         'zwadp = zwad.ad:ZtfAnomalyDetector.script'
                     ]
                 })
