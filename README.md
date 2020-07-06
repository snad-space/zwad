# zwad
ZTF anomaly detection

## Package installation
Before working with the code, the package should be installed in the development
mode:
```shell
git clone git@github.com:snad-space/zwad.git
pip install -e zwad
```

## Passive(?) anomaly detection example

```shell
# Get the data and extract it right here
wget http://sai.snad.space/ztf/m31.tar.gz
tar xzf m32.tar.gz

# Run one algorithm
zwadp -c iso oid_m31.dat feature_m31.dat > m31_iso.csv

# Run a few more algorithms
for ALGO in iso lof gmm svm; do
  zwadp -c ${ALGO} oid_m31.dat feature_m31.dat > m31_${ALGO}.csv
done
```
