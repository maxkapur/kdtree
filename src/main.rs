use num::Float;
use rand::prelude::*;
use std::rc::Rc;

const MAX_N_SAMPLES: usize = 100;
const N_POINTS_FAKEDATA: usize = 1000000;

fn main() {
    let points = fake_data(None);
    for i in 0..25 {
        println!("point: {:?}", points[i])
    }
    let tree = KDTree::new(&points, None);
    println!("tree created");
    let p1 = points[0];
    let p2 = [0.3, 0.1, 0.7];

    println!("contains p1: {}", tree.contains(&p1));
    println!("contains p2: {}", tree.contains(&p2));
    let (neighbor, distance) = tree.nearest_neighbor(&p2, None);
    println!("nearest neighbor to p2: {:?}", neighbor);
    println!("distance: {}", distance);
}

struct Stem<T: KDTreeableFloat, const K: usize> {
    k: usize,
    median: T,
    left: KDTree<T, K>,
    right: KDTree<T, K>,
}

struct Leaf<T: KDTreeableFloat, const K: usize>([T; K]);

enum KDTree<T: KDTreeableFloat, const K: usize> {
    Stem(Rc<Stem<T, K>>),
    Leaf(Rc<Leaf<T, K>>),
}

impl<T: KDTreeableFloat, const K: usize> KDTree<T, K> {
    pub fn new(points: &Vec<[T; K]>, k: Option<usize>) -> KDTree<T, K> {
        if points.len() == 1 {
            return KDTree::Leaf(Leaf(points[0]).into());
        }

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

        let left = KDTree::new(&left_points, Some((k + 1) % K));
        let right = KDTree::new(&right_points, Some((k + 1) % K));

        return KDTree::Stem(
            Stem {
                k,
                median,
                left,
                right,
            }
            .into(),
        );
    }

    pub fn contains(&self, point: &[T; K]) -> bool {
        // If you think about it, just running the nearest neighbor search
        // and checking for exact match is basically fastest way to do this,
        // because we want to use the best distance so far to branch and bound
        // even in the search case.
        let (nearest_point, distance) = self.nearest_neighbor(point, None);
        return distance <= 0.0.into();
    }

    pub fn nearest_neighbor(
        &self,
        point: &[T; K],
        best_so_far: Option<([T; K], T)>,
    ) -> ([T; K], T) {
        let mut best_so_far: ([T; K], T) =
            best_so_far.unwrap_or(([f64::nan().into(); K], f64::infinity().into()));
        match self {
            KDTree::Leaf(leaf) => {
                return closer_of(point, leaf.0.into(), best_so_far);
            }
            KDTree::Stem(stem) => {
                // Any points in my left, right arm have distance at least
                let (min_left, min_right) = min_lr(point[stem.k], stem.median);

                if min_left < best_so_far.1 {
                    best_so_far = stem.left.nearest_neighbor(point, Some(best_so_far))
                }
                if min_right < best_so_far.1 {
                    best_so_far = stem.right.nearest_neighbor(point, Some(best_so_far))
                }

                return best_so_far;
            }
        }
    }

    // todo: maybe
    pub fn push() {}

    // todo: maybe
    pub fn collect() {}

    // todo: maybe
    pub fn merge() {}
}

// The minimum distance achievable in the left and right arms
// of a node when the current value and median are as given
fn min_lr<T: KDTreeableFloat>(value: T, median: T) -> (T, T) {
    let mut diff = median - value;
    diff = diff * diff;
    return if value <= median {
        (0.0.into(), diff)
    } else {
        (diff, 0.0.into())
    };
}

fn squared_distance<T: KDTreeableFloat, const K: usize>(point0: &[T; K], point1: &[T; K]) -> T {
    let init: T = 0.0.into();
    return (0..K).fold(init, |accum, i| {
        let diff = point0[i] - point1[i];
        accum + diff * diff
    });
}

fn closer_of<T: KDTreeableFloat, const K: usize>(
    point: &[T; K],
    candidate_point: [T; K],
    best_so_far: ([T; K], T),
) -> ([T; K], T) {
    let candidate_distance = squared_distance(point, &candidate_point);
    return if best_so_far.1 <= candidate_distance {
        best_so_far
    } else {
        (candidate_point, candidate_distance)
    };
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
