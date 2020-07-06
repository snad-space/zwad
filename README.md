# zwad
ZTF anomaly detection

## Package installation
Before working with the code, the package should be installed in the development
mode:
```shell
git clone git@github.com:snad-space/zwad.git
cd zwad
pip install -e .
```

## Data download
```shell
cd data
for FILE in deep disk m31; do
    wget "http://sai.snad.space/ztf/${FILE}.tar.gz" -O - | tar -zxf -
done
```

## Passive(?) anomaly detection example

```shell
# Run one algorithm
zwadp -c iso data/oid_m31.dat data/feature_m31.dat > data/m31_iso.csv

# zwadp uses only one core by default. Number of parallel
# jobs may be increased with the -j option.
zwadp -c iso -j 4 data/oid_m31.dat data/feature_m31.dat > data/m31_iso.csv

# Run a few more algorithms
for ALGO in iso lof gmm svm; do
  zwadp -c ${ALGO} data/oid_m31.dat data/feature_m31.dat > data/m31_${ALGO}.csv
done

# See the help
zwadp -h
```
