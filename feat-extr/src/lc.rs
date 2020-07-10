pub const MJD0: f64 = 58000.0;

#[derive(Clone)]
pub struct LightCurveEntry {
    pub oid: u64,
    pub t: Vec<f32>,
    pub mag: Vec<f32>,
    pub err2: Vec<f32>,
}

pub struct LightCurveObservation {
    pub oid: u64,
    pub t: f32,
    pub mag: f32,
    pub err2: f32,
}
