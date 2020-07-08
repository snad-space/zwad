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
                     'ad_examples @ git+https://github.com/shubhomoydas/ad_examples.git#egg=ad_examples-0.0.0'],
                 entry_points={
                     'console_scripts': [
                         'zwaad = zwad.aad:execute_from_commandline',
                         'zwadp = zwad.ad:ZtfAnomalyDetector.script'
                     ]
                 })
