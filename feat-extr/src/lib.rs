use light_curve_common::linspace;
use light_curve_feature::*;
use light_curve_interpol::Interpolator;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub mod ch;
use ch::CHLightCurves;

pub mod config;
use config::{Config, DataBase};

mod dump;
use dump::Dumper;

mod hdf;
use hdf::Hdf5Cache;

mod lc;

mod pg;
use pg::PGLightCurves;

mod traits;
use traits::{Cache, LightCurvesDataBase, ObservationsToLightCurves};

struct TruncMedianNyquistFreq {
    m: MedianNyquistFreq,
    max_freq: f32,
}

impl TruncMedianNyquistFreq {
    fn new(min_dt: f32) -> Self {
        Self {
            m: MedianNyquistFreq,
            max_freq: std::f32::consts::PI / min_dt,
        }
    }
}

impl NyquistFreq<f32> for TruncMedianNyquistFreq {
    fn nyquist_freq(&self, t: &[f32]) -> f32 {
        f32::min(self.m.nyquist_freq(t), self.max_freq)
    }
}

struct TruncQuantileNyquistFreq {
    q: QuantileNyquistFreq,
    max_freq: f32,
}

impl TruncQuantileNyquistFreq {
    fn new(quantile: f32, min_dt: f32) -> Self {
        Self {
            q: QuantileNyquistFreq { quantile },
            max_freq: std::f32::consts::PI / min_dt,
        }
    }
}

impl NyquistFreq<f32> for TruncQuantileNyquistFreq {
    fn nyquist_freq(&self, t: &[f32]) -> f32 {
        f32::min(self.q.nyquist_freq(t), self.max_freq)
    }
}

struct MaxNyquistFreq;

impl NyquistFreq<f32> for MaxNyquistFreq {
    fn nyquist_freq(&self, t: &[f32]) -> f32 {
        let dt = (0..t.len() - 1)
            .map(|i| t[i + 1] - t[i])
            // .filter(|&x| x > 0.0)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        core::f32::consts::PI / dt
    }
}

pub fn run(config: Config) {
    let mut dumper = Dumper::new();

    if let Some(oid_path) = config.oid_path {
        dumper.set_oid_writer(oid_path.clone());
    }

    if let Some(ic) = &config.interpolation_config {
        let interpolator = Interpolator {
            target_x: linspace(58194.5_f32, 58482.5, 145),
            left: 0.,
            right: 0.,
        };
        dumper.set_interpolator(ic.path.clone(), interpolator);
    }

    if let Some(fc) = &config.feature_config {
        let mut periodogram_feature_evaluator = Periodogram::new(3);
        // periodogram_feature_evaluator
        //     .set_nyquist(Box::new(TruncQuantileNyquistFreq::new(0.05, 60.0 / 86400.0)));
        periodogram_feature_evaluator
            .set_nyquist(Box::new(TruncMedianNyquistFreq::new(300.0 / 86400.0)));
        periodogram_feature_evaluator.set_max_freq_factor(2.0);
        periodogram_feature_evaluator.add_features(vec![
            Box::new(Amplitude::default()),
            Box::new(BeyondNStd::default()),
            Box::new(BeyondNStd::new(2.0)),
            Box::new(Cusum::default()),
            Box::new(Eta::default()),
            Box::new(InterPercentileRange::default()),
            Box::new(StandardDeviation::default()),
            Box::new(PercentAmplitude::default()),
        ]);
        let feature_extractor = feat_extr!(
            Amplitude::default(),
            BeyondNStd::default(),
            BeyondNStd::new(2.0),
            Cusum::default(),
            Eta::default(),
            EtaE::default(),
            InterPercentileRange::default(),
            InterPercentileRange::new(0.1),
            Kurtosis::default(),
            LinearFit::default(),
            LinearTrend::default(),
            MagnitudePercentageRatio::new(0.4, 0.05), // default
            MagnitudePercentageRatio::new(0.2, 0.1),
            MaximumSlope::default(),
            Mean::default(),
            MedianAbsoluteDeviation::default(),
            MedianBufferRangePercentage::new(0.05), // not default
            PercentAmplitude::default(),
            PercentDifferenceMagnitudePercentile::new(0.05), // default
            PercentDifferenceMagnitudePercentile::new(0.2),
            periodogram_feature_evaluator,
            ReducedChi2::default(),
            Skew::default(),
            StandardDeviation::default(),
            StetsonK::default(),
            WeightedMean::default(),
        );
        dumper.set_feature_extractor(
            fc.value_path.clone(),
            fc.name_path.clone(),
            feature_extractor,
        );
    }

    let read_cache = match &config.cache_config {
        Some(cc) => {
            let cache = Box::new(Hdf5Cache {
                path: cc.data_path.clone(),
            });
            if Path::new(&cc.query_path).exists() {
                let query_cache = std::fs::read(&cc.query_path).unwrap();
                let query_from_file = String::from_utf8_lossy(&query_cache);
                assert_eq!(
                    query_from_file, config.sql_query,
                    "Cached SQL query mismatched specified one"
                );
                Some(cache)
            } else {
                let mut query_file = File::create(cc.query_path.clone()).unwrap();
                write!(query_file, "{}", config.sql_query).unwrap();
                dumper.set_write_cache(cache);
                None
            }
        }
        None => None,
    };

    match read_cache {
        Some(cache) => {
            dumper.dump_query_iter(cache.reader());
        }
        None => match config.database {
            DataBase::Postgres => {
                let mut light_curves = PGLightCurves::new(&config.connection_config);
                let lc_query = light_curves.query(&config.sql_query);
                let lce_iter = lc_query
                    .into_iter()
                    .light_curves(config.light_curves_are_sorted);
                dumper.dump_query_iter(lce_iter);
            }
            DataBase::ClickHouse => {
                let mut light_curves = CHLightCurves::new(&config.connection_config);
                let lc_query = light_curves.query(&config.sql_query);
                let lce_iter = lc_query
                    .into_iter()
                    .light_curves(config.light_curves_are_sorted);
                dumper.dump_query_iter(lce_iter);
            }
        },
    }

    dumper.write_names();
}
