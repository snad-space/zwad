#### Source code for
# Anomaly detection in the Zwicky Transient Facility DR3

<a href="https://ascl.net/2106.033"><img src="https://img.shields.io/badge/ascl-2106.033-blue.svg?colorB=262255" alt="ascl:2106.033" /></a> ![GithHub actions badge](https://github.com/snad-space/zwad/actions/workflows/python.yml/badge.svg)

This guide allows you to use the ZTF anomaly detection tool presented by [*Malanchev et al., 2020*](https://arxiv.org/abs/2012.01419).

## Package installation

Install current version by
```shell
pip install git+https://github.com/snad-space/zwad
```

### Development
Before working with the code, the package should be installed in the development
mode:
```shell
git clone git@github.com:snad-space/zwad.git
cd zwad
pip install -e .
```

## Data download
Light curve feature data for ZTF DR3 fields used in the research are available on [Zenodo](https://zenodo.org/record/4318700).
You can download it from the website or by:
```shell
cd data
zwad-zenodo
```

## Example

```shell
# Run one algorithm
zwadp -c iso --oid oid_m31.dat --feature feature_m31.dat > m31_iso.csv

# zwadp uses only one core by default. Number of parallel
# jobs may be increased with the -j option.
zwadp -c iso -j 4 --oid oid_m31.dat --feature feature_m31.dat > m31_iso.csv

# Run a few more algorithms
for ALGO in iso lof gmm svm; do
  zwadp -c ${ALGO} --oid oid_m31.dat --feature feature_m31.dat > m31_${ALGO}.csv
done

# Combine data-sets:
zwadp -c iso --oid oid_m31.dat --oid fakes/oid_m31_fake.dat --feature feature_m31.dat --feature fakes/feature_m31_fake.dat > m31_iso_fake.csv

# See the help
zwadp -h
```

### Active anomaly detection

Lets use the same data from [Zenodo](https://zenodo.org/record/4318700).
```shell
cd data
zwad-zenodo
```

It is better to create temporary directory to run active anomaly detection:
```shell
mkdir ../tmp
cd tmp
```

```shell
# Run the algorithm with M31 dataset
zwaad --oid ../data/oid_m31.dat --feature ../data/feature_m31.dat --feature-names ../data/feature_m31.name --budget 4
```

The script output may be as the following:
```
Check https://ztf.snad.space/dr4/view/695211400124577 for details
Is 695211400124577 anomaly? [y/N]:
```
It is waiting for our decision now. Let us decide that `695211400124577` is an anomaly and continue:
```
Check https://ztf.snad.space/dr4/view/695211400124577 for details
Is 695211400124577 anomaly? [y/N]: y
Check https://ztf.snad.space/dr4/view/695211400034403 for details
Is 695211400034403 anomaly? [y/N]: n
Check https://ztf.snad.space/dr4/view/695211400070144 for details
Is 695211400070144 anomaly? [y/N]: n
Check https://ztf.snad.space/dr4/view/695211400134512 for details
Is 695211400134512 anomaly? [y/N]: y
UNK  4
```

Now we have examined all four samples within our budget.
```shell
# Check found anomalies
cat ./data/aad/anomalies_feature_m31.txt
695211400124577
695211400134512

# Check answers log
cat ./data/aad/answers_feature_m31.csv
oid,is_anomaly
695211400124577,1
695211400034403,0
695211400070144,0
695211400134512,1
```

## Related repositories
- [SNAD ZTF DR web-viewer](https://github.com/snad-space/ztf-viewer)
- [Light-curve features](https://github.com/hombit/light-curve)
- [SNAD ZTF DR API](https://github.com/snad-space/snad-ztf-db)
- [SNAD Light-curve feature API](https://github.com/snad-space/web-light-curve-features)
- [SNAD OGLE-III metadata API](https://github.com/snad-space/snad-ogle3)
- [SNAD ZTF Periodic Catalog API](https://github.com/snad-space/ztf-periodic-catalog-db)
