use rand::prelude::*;
use num::Float;

const MAX_N_SAMPLES: usize = 100;
const N_POINTS_FAKEDATA: usize = 1000;

fn main() {
    let mut v = vec![0.7, 0.8, 0.1, -0.5, 0.11];

    let points = fake_data(None);
    let median = median_x(&points);
    println!("median_x points: {median:?}");


}

fn median_x<T: KDTreeableFloat>(points: &Vec<Vec3D<T>>) -> T {
    let mut xs = points.iter().map(|point| point.x).collect::<Vec<T>>();
    return sample_median(&mut xs, None);
}



trait KDTreeableFloat: PartialOrd + Float + From<f64> {}
impl<T: PartialOrd + Float + From<f64>> KDTreeableFloat for T {}


#[derive(Debug)]
struct Vec3D<T> {
    x: T,
    y: T,
    z: T,
}

fn fake_data(n_points: Option<usize>) -> Vec<Vec3D<f64>> {
    let n_points = n_points.unwrap_or(N_POINTS_FAKEDATA);
    return (0..n_points)
        .map(|_| {
            Vec3D {
                x: rand::random::<f64>(),
                y: rand::random::<f64>(),
                z: rand::random::<f64>(),
            }
        })
        .collect::<Vec<Vec3D<f64>>>();
}


fn sample_median<T: KDTreeableFloat>(v: &mut Vec<T>, max_n_samples: Option<usize>) -> T {
    let max_n_samples: usize = max_n_samples.unwrap_or(MAX_N_SAMPLES);
    let mut rng = rand::thread_rng();
    let (shuffled, _) = v.partial_shuffle(&mut rng, max_n_samples);

    shuffled.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = shuffled.len();
    return match len % 2 {
        0 => (shuffled[len / 2] + shuffled[len / 2 + 1]) / 2.0.into(),
        1 => shuffled[len / 2],
        _ => panic!("{}", len % 2),
    };
}
