# coding=utf-8
import setuptools

setuptools.setup(name='zwad',
                 license_files = ('LICENSE',),
                 license='MIT',
                 packages=['zwad'],
                 install_requires=[
                     'coniferest @ git+https://github.com/snad-space/coniferest.git@de3112ab0f2b5bea5d7d33887b2e119b1476840e#egg=coniferest-0.0.2',
                     'pandas<2.0',
                     'click',
                     'pillow',
                     'scikit-learn>=0.23',
                     'seaborn',
                     'tqdm',
                     'matplotlib>=3.1,<4.0',
                     'astropy',
                     'requests>=2,<3.0',
                     'onnx',
                     'ad_examples @ git+https://github.com/shubhomoydas/ad_examples.git@0a7b86c4f2f7306ff543a15b387fe938f9c06130#egg=ad_examples-0.0.1',
                     'aad2onnx'
                 ],
                 entry_points={
                     'console_scripts': [
                         'zwaad = zwad.aad:execute_from_commandline',
                         'zwadp = zwad.ad:ZtfAnomalyDetector.script',
                         'zwann = zwad.nn:execute_from_commandline',
                         'zwad-zenodo = zwad.download_feature_set:execute_from_commandline',
                     ]
                 })
