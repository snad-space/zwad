use crate::lc::{LightCurveObservation, MJD0};
use crate::traits::{LightCurvesDataBase, ObservationsToLightCurves};
use async_std::task;
use clickhouse_rs::errors::Error;
use clickhouse_rs::types::{Block, FromSql};
use clickhouse_rs::{ClientHandle, Pool};
use futures_util::stream::{BoxStream, StreamExt};

pub struct CHLightCurves {
    client: ClientHandle,
}

impl CHLightCurves {
    pub fn new(url: &str) -> Self {
        let pool = Pool::new(url);
        let client = task::block_on(pool.get_handle()).unwrap();
        Self { client }
    }
}

impl<'a> LightCurvesDataBase<'a> for CHLightCurves {
    type Query = CHLightCurvesQuery<'a>;

    fn query(&'a mut self, query: &str) -> Self::Query {
        CHLightCurvesQuery::new(self, query)
    }
}

pub struct CHLightCurvesQuery<'a> {
    stream: BoxStream<'a, Result<Block, Error>>,
}

impl<'a> CHLightCurvesQuery<'a> {
    pub fn new(ch_light_curves: &'a mut CHLightCurves, query: &str) -> Self {
        let stream = ch_light_curves.client.query(query).stream_blocks();
        Self { stream }
    }
}

impl<'a> IntoIterator for CHLightCurvesQuery<'a> {
    type Item = LightCurveObservation;
    type IntoIter = CHLightCurvesQueryIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

struct Row<'b> {
    block: &'b Block,
    idx: usize,
}

impl<'b> Row<'b> {
    fn get<T>(&self, col: &str) -> Result<T, Error>
    where
        T: FromSql<'b>,
    {
        self.block.get(self.idx, col)
    }
}

pub struct CHLightCurvesQueryIterator<'a> {
    query: CHLightCurvesQuery<'a>,
    block: Block,
    block_size: usize,
    row_block_idx: usize,
}

impl<'a> CHLightCurvesQueryIterator<'a> {
    fn new(mut query: CHLightCurvesQuery<'a>) -> Self {
        let block = task::block_on(query.stream.next()).unwrap().unwrap();
        let block_size = block.row_count();
        Self {
            query,
            block,
            block_size,
            row_block_idx: 0,
        }
    }

    fn row_to_obs(row: Row) -> LightCurveObservation {
        let oid: u64 = row.get("oid").unwrap();
        let mjd: f64 = row.get("mjd").unwrap();
        let t = (mjd - MJD0) as f32;
        let mag: f32 = row.get("mag").unwrap();
        let magerr: f32 = row.get("magerr").unwrap();
        let err2 = magerr.powi(2);
        LightCurveObservation { oid, t, mag, err2 }
    }
}

impl<'a> Iterator for CHLightCurvesQueryIterator<'a> {
    type Item = LightCurveObservation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_block_idx < self.block_size {
            self.row_block_idx += 1;
        } else {
            self.block_size = 0;
            while self.block_size == 0 {
                let block = task::block_on(self.query.stream.next());
                if block.is_none() {
                    return None;
                }
                self.block = block.unwrap().unwrap();
                self.block_size = self.block.row_count();
            }
            self.row_block_idx = 1;
        }
        Some(Self::row_to_obs(Row {
            block: &self.block,
            idx: self.row_block_idx - 1,
        }))
    }
}

impl<'a> ObservationsToLightCurves for CHLightCurvesQueryIterator<'a> {}
