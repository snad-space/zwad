use crate::lc::{LightCurveObservation, MJD0};
use crate::traits::{LightCurvesDataBase, ObservationsToLightCurves};
use fallible_iterator::FallibleIterator;
use fallible_iterator::Iterator as NonFallibleIterator;
use postgres::{Client, NoTls, RowIter};
use std::iter;

pub struct PGLightCurves {
    client: Client,
}

impl PGLightCurves {
    pub fn new(params: &str) -> Self {
        Self {
            client: Client::connect(params, NoTls).expect("Cannot connect to Postgres"),
        }
    }
}

impl<'a> LightCurvesDataBase<'a> for PGLightCurves {
    type Query = PGLightCurvesQuery<'a>;

    fn query(&'a mut self, query: &str) -> Self::Query {
        PGLightCurvesQuery::new(self, query)
    }
}

pub struct PGLightCurvesQuery<'a> {
    query_iter: NonFallibleIterator<RowIter<'a>>,
}

impl<'a> PGLightCurvesQuery<'a> {
    pub fn new(pg_light_curves: &'a mut PGLightCurves, query: &str) -> Self {
        let query_iter = pg_light_curves
            .client
            .query_raw(query, iter::empty())
            .expect("Postgres query execution error")
            .iterator();
        Self { query_iter }
    }
}

impl<'a> IntoIterator for PGLightCurvesQuery<'a> {
    type Item = LightCurveObservation;
    type IntoIter = PGLightCurvesQueryIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PGLightCurvesQueryIterator::new(self)
    }
}

pub struct PGLightCurvesQueryIterator<'a> {
    query: Option<PGLightCurvesQuery<'a>>,
}

impl<'a> PGLightCurvesQueryIterator<'a> {
    fn new(query: PGLightCurvesQuery<'a>) -> Self {
        Self { query: Some(query) }
    }
}

impl<'a> Iterator for PGLightCurvesQueryIterator<'a> {
    type Item = LightCurveObservation;

    fn next(&mut self) -> Option<Self::Item> {
        // query_iter returns Error x when it should be empty
        let x = match self.query.as_mut() {
            Some(query) => match query.query_iter.next() {
                Some(x) => x,
                None => {
                    self.query = None;
                    return None;
                }
            },
            None => return None,
        };
        let row = x.expect("Error on reading from database");
        let oid: i64 = row.get("oid");
        let oid = oid as u64;
        let mjd: f32 = row.get("mjd");
        let t = mjd - (MJD0 as f32);
        let mag = row.get("mag");
        let err: f32 = row.get("magerr");
        let err2 = err.powi(2);
        Some(LightCurveObservation { oid, t, mag, err2 })
    }
}

impl<'a> ObservationsToLightCurves for PGLightCurvesQueryIterator<'a> {}
