#!/bin/bash

docker-compose run --rm clickhouse_cyg \
    clickhouse \
    'SELECT oid, mjd, mag, magerr
        FROM ztf.dr3
        INNER JOIN
            (SELECT h3index10, oid FROM ztf.dr3_meta_short WHERE fieldid = 795 AND filter = 2 AND ngoodobs >= 100) AS meta
            USING (h3index10, oid)
        WHERE fieldid = 795 AND catflags = 0 AND mjd <= 58483.0
        ORDER BY h3index10, oid, mjd
' \
    --suffix=_deep \
    --connect="tcp://api@snad.sai.msu.ru:9000/ztf" \
    --sorted \
    --features
