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

struct CurrentBlock {
    block: Block,
    size: usize,
    idx: usize,
}

impl CurrentBlock {
    fn new(block: Block) -> Self {
        let size = block.row_count();
        Self {
            block,
            size,
            idx: 0,
        }
    }
}

pub struct CHLightCurvesQueryIterator<'a> {
    query: CHLightCurvesQuery<'a>,
    block: Option<CurrentBlock>,
}

impl<'a> CHLightCurvesQueryIterator<'a> {
    fn new(query: CHLightCurvesQuery<'a>) -> Self {
        Self { query, block: None }
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
        while self.block.is_none()
            || self.block.as_ref().unwrap().size == self.block.as_ref().unwrap().idx
        {
            match task::block_on(self.query.stream.next()) {
                Some(block) => self.block = Some(CurrentBlock::new(block.unwrap())),
                None => return None,
            }
        }

        match &mut self.block {
            Some(cur_block) => {
                cur_block.idx += 1;
                Some(Self::row_to_obs(Row {
                    block: &cur_block.block,
                    idx: cur_block.idx - 1,
                }))
            }
            None => panic!("We cannot be here"),
        }
    }
}

impl<'a> ObservationsToLightCurves for CHLightCurvesQueryIterator<'a> {}
