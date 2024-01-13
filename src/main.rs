use rand::prelude::*;
use num::Float;

const MAX_N_SAMPLES: usize = 10;

fn main() {
    println!("Hello, world!");

    let mut v = vec![0.7, 0.8, 0.1, -0.5, 0.11];

    let med = sample_median(&mut v, None);
    println!("v is {:?}", v);
    println!("sample median is {med:?}")
}

fn sample_median<T: PartialOrd + Float + From<f64>>(v: &mut Vec<T>, max_n_samples: Option<usize>) -> T
{
    let max_n_samples: usize = max_n_samples.unwrap_or(MAX_N_SAMPLES);
    let mut rng = rand::thread_rng();
    let (shuffled, _) = v.partial_shuffle(&mut rng, max_n_samples);

    shuffled.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = shuffled.len();
    return match len % 2 {
        0 => (shuffled[len / 2] + shuffled[len / 2 + 1]) / 2.0.into(),
        1 => shuffled[len / 2],
        _ => panic!("{}", len % 2)
    };
}

