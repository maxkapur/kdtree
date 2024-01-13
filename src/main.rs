use num::Float;
use rand::prelude::*;

const MAX_N_SAMPLES: usize = 100;
const N_POINTS_FAKEDATA: usize = 1000;

fn main() {
    let points = fake_data(None);
    for i in 0..25 {
        println!("point: {:?}", points[i])
    }
    let tree = KDSubtree::new(&points, None);
}

enum Node<T: KDTreeableFloat, const K: usize> {
    Empty,
    Leaf(Box<[T; K]>),
    Stem(Box<KDSubtree<T, K>>),
}

impl<T: KDTreeableFloat, const K: usize> Node<T, K> {
    fn new(points: &Vec<[T; K]>, k: usize) -> Node<T, K> {
        if points.is_empty() {
            return Node::Empty;
        };

        let (first, rest) = points.split_first().unwrap();
        if rest.is_empty() {
            return Node::Leaf(Box::new(*first));
        };

        return Node::Stem(Box::new(KDSubtree::new(points, Some(k))));
    }
}

struct KDSubtree<T: KDTreeableFloat, const K: usize> {
    median: T,
    left: Node<T, K>,
    right: Node<T, K>,
}

impl<T: KDTreeableFloat, const K: usize> KDSubtree<T, K> {
    pub fn new(points: &Vec<[T; K]>, k: Option<usize>) -> KDSubtree<T, K> {
        let k = k.unwrap_or(0);
        let mut v = points.iter().map(|point| point[k]).collect::<Vec<T>>();

        let median = sample_median(&mut v, None);
        let mut left_points = Vec::<[T; K]>::with_capacity(points.len() / 2);
        let mut right_points = Vec::<[T; K]>::with_capacity(points.len() / 2);
        for point in points {
            if point[k] <= median {
                left_points.push(*point);
            } else {
                right_points.push(*point);
            }
        }

        let left = Node::new(&left_points, (k + 1) % K);
        let right = Node::new(&right_points, (k + 1) % K);

        return KDSubtree {
            median,
            left,
            right,
        };
    }

    // todo:
    pub fn contains() {}

    // todo:
    pub fn nearest_neighbor() {}

    // todo: maybe
    pub fn push() {}

    // todo: maybe
    pub fn collect() {}

    // todo: maybe
    pub fn merge() {}
}

trait KDTreeableFloat: PartialOrd + Float + From<f64> {}
impl<T: PartialOrd + Float + From<f64>> KDTreeableFloat for T {}

fn fake_data(n_points: Option<usize>) -> Vec<[f64; 3]> {
    let n_points = n_points.unwrap_or(N_POINTS_FAKEDATA);
    return (0..n_points)
        .map(|_| {
            [
                rand::random::<f64>(),
                rand::random::<f64>(),
                rand::random::<f64>(),
            ]
        })
        .collect::<Vec<[f64; 3]>>();
}

fn sample_median<T: KDTreeableFloat>(v: &mut Vec<T>, max_n_samples: Option<usize>) -> T {
    let max_n_samples: usize = max_n_samples.unwrap_or(MAX_N_SAMPLES);
    let mut rng = rand::thread_rng();
    let (shuffled, _) = v.partial_shuffle(&mut rng, max_n_samples);

    shuffled.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = shuffled.len();
    return match len % 2 {
        0 => (shuffled[len / 2 - 1] + shuffled[len / 2]) / 2.0.into(),
        1 => shuffled[len / 2],
        _ => panic!("{}", len % 2),
    };
}
