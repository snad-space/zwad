use crate::lc::{LightCurveEntry, LightCurveObservation};
use dyn_clone::DynClone;
use unzip3::Unzip3;

pub trait LightCurvesDataBase<'a> {
    type Query: IntoIterator<Item = LightCurveObservation>;

    fn query(&'a mut self, query: &str) -> Self::Query;
}

pub trait Dump: Send + Sync {
    fn eval(&self, lce: &LightCurveEntry) -> Vec<u8>;
    fn get_names(&self) -> Vec<&str>;
    fn get_value_path(&self) -> &str;
    fn get_name_path(&self) -> Option<&str>;
}

pub trait Cache: DynClone + Send {
    fn reader(&self) -> Box<dyn Iterator<Item = LightCurveEntry>>;
    fn writer(&self) -> Box<dyn CacheWriter>;
}

pub trait CacheWriter {
    fn write(&mut self, lce: &LightCurveEntry);
}

pub trait ObservationsToLightCurves: Iterator<Item = LightCurveObservation>
where
    Self: Sized,
{
    fn light_curves(self, sorted: bool) -> LightCurveIterator<Self> {
        LightCurveIterator::new(self, sorted)
    }
}

pub struct LightCurveIterator<I>
where
    I: Iterator<Item = LightCurveObservation>,
{
    observations: I,
    sorted: bool,
    current_obs: Option<LightCurveObservation>,
}

impl<I> LightCurveIterator<I>
where
    I: Iterator<Item = LightCurveObservation>,
{
    fn new(observations: I, sorted: bool) -> Self {
        Self {
            observations,
            sorted,
            current_obs: None,
        }
    }
}

impl<I> Iterator for LightCurveIterator<I>
where
    I: Iterator<Item = LightCurveObservation>,
{
    type Item = LightCurveEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let (oid, mut triples) = match self.current_obs.as_ref() {
            Some(obs) => (obs.oid, vec![(obs.t, obs.mag, obs.err2)]),
            None => {
                let next_obs = self.observations.next();
                match next_obs.as_ref() {
                    Some(obs) => (obs.oid, vec![(obs.t, obs.mag, obs.err2)]),
                    None => return None,
                }
            }
        };
        self.current_obs = None;
        while let Some(obs) = self.observations.next() {
            if obs.oid != oid {
                self.current_obs = Some(obs);
                break;
            }
            triples.push((obs.t, obs.mag, obs.err2));
        }
        if !self.sorted {
            triples.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        let (t, mag, err2) = triples.into_iter().unzip3();
        Some(LightCurveEntry { oid, t, mag, err2 })
    }
}
