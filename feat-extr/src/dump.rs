use crate::lc::LightCurveEntry;
use crate::traits::*;
use crossbeam::channel::{bounded as bounded_channel, Receiver, Sender};
use dyn_clone::clone_box;
use light_curve_feature::{time_series::TimeSeries, FeatureExtractor};
use light_curve_interpol::Interpolator;
use num_cpus;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::iter::Iterator;
use std::ops::Deref;
use std::sync::Arc;
use std::thread;

const MIN_NUMBER_OF_OBSERVATIONS: usize = 4;

struct FluxDump {
    path: String,
    interpolator: Interpolator<f32, f32>,
}

impl Dump for FluxDump {
    fn eval(&self, lce: &LightCurveEntry) -> Vec<u8> {
        let flux: Vec<_> = lce.mag.iter().map(|&x| 10_f32.powf(-0.4 * x)).collect();
        self.interpolator
            .interpolate(&lce.t[..], &flux[..])
            .iter()
            .flat_map(|x| x.to_bits().to_ne_bytes().to_vec())
            .collect()
    }

    fn get_names(&self) -> Vec<&str> {
        vec![]
    }

    fn get_value_path(&self) -> &str {
        self.path.as_str()
    }

    fn get_name_path(&self) -> Option<&str> {
        None
    }
}

struct FeatureDump {
    value_path: String,
    name_path: String,
    feature_extractor: FeatureExtractor<f32>,
}

impl Dump for FeatureDump {
    fn eval(&self, lce: &LightCurveEntry) -> Vec<u8> {
        let ts = TimeSeries::new(&lce.t[..], &lce.mag[..], Some(&lce.err2[..]));
        self.feature_extractor
            .eval(ts)
            .iter()
            .flat_map(|x| x.to_bits().to_ne_bytes().to_vec())
            .collect()
    }

    fn get_names(&self) -> Vec<&str> {
        self.feature_extractor.get_names()
    }

    fn get_value_path(&self) -> &str {
        self.value_path.as_str()
    }

    fn get_name_path(&self) -> Option<&str> {
        Some(self.name_path.as_str())
    }
}

struct OIDDump {
    path: String,
}

impl Dump for OIDDump {
    fn eval(&self, lce: &LightCurveEntry) -> Vec<u8> {
        lce.oid.to_ne_bytes().to_vec()
    }

    fn get_names(&self) -> Vec<&str> {
        vec![]
    }

    fn get_value_path(&self) -> &str {
        self.path.as_str()
    }

    fn get_name_path(&self) -> Option<&str> {
        None
    }
}

pub struct Dumper {
    dumps: Vec<Arc<dyn Dump>>,
    write_caches: Vec<Box<dyn Cache>>,
}

impl Dumper {
    pub fn new() -> Self {
        Self {
            dumps: vec![],
            write_caches: vec![],
        }
    }

    pub fn set_oid_writer(&mut self, oid_path: String) -> &mut Self {
        self.dumps.push(Arc::new(OIDDump { path: oid_path }));
        self
    }

    pub fn set_interpolator(
        &mut self,
        flux_path: String,
        interpolator: Interpolator<f32, f32>,
    ) -> &mut Self {
        self.dumps.push(Arc::new(FluxDump {
            path: flux_path,
            interpolator,
        }));
        self
    }

    pub fn set_feature_extractor(
        &mut self,
        value_path: String,
        name_path: String,
        feature_extractor: FeatureExtractor<f32>,
    ) -> &mut Self {
        self.dumps.push(Arc::new(FeatureDump {
            value_path,
            name_path,
            feature_extractor,
        }));
        self
    }

    pub fn set_write_cache(&mut self, cache: Box<dyn Cache>) -> &mut Self {
        self.write_caches.push(cache);
        self
    }

    fn writer_from_path(path: &str) -> BufWriter<File> {
        let file = File::create(path).unwrap();
        BufWriter::new(file)
    }

    fn dump_eval_worker(
        dumps: Vec<Arc<dyn Dump>>,
        receiver: Receiver<LightCurveEntry>,
        sender: Sender<Vec<Vec<u8>>>,
    ) {
        while let Ok(lce) = receiver.recv() {
            let results = dumps.iter().map(|dump| dump.eval(&lce)).collect();
            sender
                .send(results)
                .expect("Cannot send evaluation result to writer");
        }
    }

    fn dump_writer_worker(dumps: Vec<Arc<dyn Dump>>, receiver: Receiver<Vec<Vec<u8>>>) {
        let mut writers: Vec<_> = dumps
            .iter()
            .map(|dump| Self::writer_from_path(dump.get_value_path()))
            .collect();
        while let Ok(data) = receiver.recv() {
            for (x, writer) in data.iter().zip(writers.iter_mut()) {
                writer.write(&x[..]).expect("Cannot write to file");
            }
        }
    }

    fn cache_writer_worker(receiver: Receiver<LightCurveEntry>, cache: Box<dyn Cache>) {
        let mut writer = cache.writer();

        while let Ok(lce) = receiver.recv() {
            writer.write(&lce);
        }
    }

    pub fn dump_query_iter(&self, lce_iter: impl Iterator<Item = LightCurveEntry>) {
        const CHANNEL_CAP: usize = 1 << 10;
        let (dump_eval_sender, dump_eval_receiver) = bounded_channel(CHANNEL_CAP);
        let (dump_writer_sender, dump_writer_receiver) = bounded_channel(CHANNEL_CAP);
        let (cache_writer_senders, cache_writer_receivers): (Vec<_>, Vec<_>) = self
            .write_caches
            .iter()
            .map(|_| bounded_channel(CHANNEL_CAP))
            .unzip();

        let dump_eval_thread_pool: Vec<_> = (0..num_cpus::get())
            .map(|_| {
                let dumps = self.dumps.clone();
                let reciever = dump_eval_receiver.clone();
                let sender = dump_writer_sender.clone();
                thread::spawn(move || Self::dump_eval_worker(dumps, reciever, sender))
            })
            .collect();
        // Remove channel parts that are cloned and moved to workers
        drop(dump_eval_receiver);
        drop(dump_writer_sender);

        let dumps = self.dumps.clone();
        let dump_writer_thread =
            thread::spawn(move || Self::dump_writer_worker(dumps, dump_writer_receiver));

        let cache_write_thread_pool: Vec<_> = self
            .write_caches
            .iter()
            .map(|cache| clone_box(cache.deref()))
            .zip(cache_writer_receivers.into_iter())
            .map(|(cache, receiver)| {
                thread::spawn(move || Self::cache_writer_worker(receiver, cache))
            })
            .collect();

        lce_iter
            .inspect(|lce| {
                for sender in cache_writer_senders.iter() {
                    sender
                        .send(lce.clone())
                        .expect("Cannot send task to cache worker");
                }
            })
            .filter(|lce| lce.t.len() >= MIN_NUMBER_OF_OBSERVATIONS)
            // Send light curve to eval worker pool
            .for_each(|lce| {
                dump_eval_sender
                    .send(lce)
                    .expect("Cannot send task to eval worker");
            });

        // Remove sender or writer_thread will never join
        drop(dump_eval_sender);
        drop(cache_writer_senders);
        for thread in dump_eval_thread_pool {
            thread.join().expect("Dumper eval worker panicked");
        }
        dump_writer_thread
            .join()
            .expect("Dumper writer worker panicked");
        for thread in cache_write_thread_pool {
            thread.join().expect("Dumper cache writer worker panicked");
        }
    }

    pub fn write_names(&self) -> usize {
        self.dumps
            .iter()
            .filter_map(|dump| dump.get_name_path().and_then(|path| Some((dump, path))))
            .map(|(dump, path)| {
                let mut writer = Self::writer_from_path(path);
                dump.get_names()
                    .iter()
                    .map(|name| {
                        writer.write(name.as_bytes()).unwrap() + writer.write(b"\n").unwrap()
                    })
                    .sum::<usize>()
            })
            .sum()
    }
}
