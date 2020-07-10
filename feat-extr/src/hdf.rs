use crate::lc::LightCurveEntry;
use crate::traits::{Cache, CacheWriter};
use hdf5::types::VarLenArray;
use hdf5::{Dataset, H5Type};
use ndarray;

const DATASET_SIZE_STEP: hdf5::Ix = 1 << 10;
static DATASET_NAME: &'static str = "dataset";

#[derive(H5Type)]
#[repr(C)]
struct LightCurveHdf5Entry {
    oid: u64,
    t: VarLenArray<f32>,
    mag: VarLenArray<f32>,
    err2: VarLenArray<f32>,
}

trait ToHdf5 {
    fn to_hdf5(&self) -> LightCurveHdf5Entry;
}

impl ToHdf5 for LightCurveEntry {
    fn to_hdf5(&self) -> LightCurveHdf5Entry {
        LightCurveHdf5Entry {
            oid: self.oid,
            t: VarLenArray::from_slice(&self.t),
            mag: VarLenArray::from_slice(&self.mag),
            err2: VarLenArray::from_slice(&self.err2),
        }
    }
}

trait FromHdf5 {
    fn from_hdf5(entry: &LightCurveHdf5Entry) -> Self;
}

impl FromHdf5 for LightCurveEntry {
    fn from_hdf5(entry: &LightCurveHdf5Entry) -> Self {
        Self {
            oid: entry.oid,
            t: entry.t.as_slice().to_vec(),
            mag: entry.mag.as_slice().to_vec(),
            err2: entry.err2.as_slice().to_vec(),
        }
    }
}

#[derive(Clone)]
pub struct Hdf5Cache {
    pub path: String,
}

impl Hdf5Cache {
    fn dataset(&self, path: String, create: bool) -> hdf5::Result<hdf5::Dataset> {
        let dataset = if create {
            let file = hdf5::File::create(path)?;
            file.new_dataset::<LightCurveHdf5Entry>()
                .resizable(true)
                .create(DATASET_NAME, [DATASET_SIZE_STEP])?
        } else {
            let file = hdf5::File::open(path)?;
            file.dataset(DATASET_NAME)?
        };
        Ok(dataset)
    }
}

impl Cache for Hdf5Cache {
    fn reader(&self) -> Box<dyn Iterator<Item = LightCurveEntry>> {
        let dataset = self.dataset(self.path.clone(), false).unwrap();
        Box::new(Hdf5CacheReader::new(dataset))
    }

    fn writer(&self) -> Box<dyn CacheWriter> {
        let dataset = self.dataset(self.path.clone(), true).unwrap();
        Box::new(Hdf5CacheWriter::new(dataset))
    }
}

struct Hdf5CacheReader {
    dataset: Dataset,
    index: usize,
    size: usize,
}

impl Hdf5CacheReader {
    fn new(dataset: Dataset) -> Self {
        let size = dataset.size();
        Self {
            dataset,
            index: 0,
            size,
        }
    }
}

impl Iterator for Hdf5CacheReader {
    type Item = LightCurveEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.index == self.size {
            None
        } else {
            let slice = ndarray::s![self.index..self.index + 1];
            self.index += 1;
            let array = self.dataset.as_reader().read_slice_1d(&slice).unwrap();
            let entry = array.first().unwrap();
            Some(LightCurveEntry::from_hdf5(entry))
        };
        result
    }
}

struct Hdf5CacheWriter {
    dataset: Dataset,
    index: usize,
    size: usize,
}

impl Hdf5CacheWriter {
    fn new(dataset: Dataset) -> Self {
        let size = dataset.size();
        Self {
            dataset,
            index: 0,
            size,
        }
    }
}

impl CacheWriter for Hdf5CacheWriter {
    fn write(&mut self, lce: &LightCurveEntry) {
        if self.index >= self.size {
            self.size += DATASET_SIZE_STEP;
            self.dataset.resize(self.size).unwrap();
        }
        let slice = ndarray::s![self.index..self.index + 1];
        self.dataset.write_slice(&[lce.to_hdf5()], &slice).unwrap();
        self.index += 1;
    }
}

impl Drop for Hdf5CacheWriter {
    fn drop(&mut self) {
        self.dataset.resize(self.index + 1).unwrap();
    }
}
