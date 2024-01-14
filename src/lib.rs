use num::Float;
use rand::prelude::*;
use std::rc::Rc;

// When calculating the median, sample (randomly) at least
// this many points to reduce time required to build the tree
const MAX_N_SAMPLES: usize = 100;

trait FriendlyFloat: PartialOrd + Float + From<f64> {}
impl<T: PartialOrd + Float + From<f64>> FriendlyFloat for T {}

struct Stem<T: FriendlyFloat, const K: usize> {
    k: usize,
    median: T,
    left: KDTree<T, K>,
    right: KDTree<T, K>,
}

struct Leaf<T: FriendlyFloat, const K: usize>([T; K]);

enum KDTree<T: FriendlyFloat, const K: usize> {
    Stem(Rc<Stem<T, K>>),
    Leaf(Rc<Leaf<T, K>>),
}

impl<T: FriendlyFloat, const K: usize> KDTree<T, K> {
    pub fn new(points: &Vec<[T; K]>) -> KDTree<T, K> {
        return KDTree::new_at_depth(&points, None);
    }

    pub fn new_at_depth(points: &Vec<[T; K]>, k: Option<usize>) -> KDTree<T, K> {
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

        let left = KDTree::new_at_depth(&left_points, Some((k + 1) % K));
        let right = KDTree::new_at_depth(&right_points, Some((k + 1) % K));

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
        match self {
            KDTree::Leaf(leaf) => {
                let candidate: [T; K] = leaf.0.into();
                return &candidate == point;
            }
            KDTree::Stem(stem) => {
                let left_side = point[stem.k] <= stem.median;
                return if left_side {
                    stem.left.contains(&point)
                } else {
                    stem.right.contains(&point)
                };
            }
        }
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
}

/// The minimum squared distance achievable in the left and right
/// arms of a node when the current value and median are as given
fn min_lr<T: FriendlyFloat>(value: T, median: T) -> (T, T) {
    let mut diff = median - value;
    diff = diff * diff;
    return if value <= median {
        (0.0.into(), diff)
    } else {
        (diff, 0.0.into())
    };
}

// Square of the Euclidean distance between two points
fn squared_distance<T: FriendlyFloat, const K: usize>(point0: &[T; K], point1: &[T; K]) -> T {
    let init: T = 0.0.into();
    return (0..K).fold(init, |accum, i| {
        let diff = point0[i] - point1[i];
        accum + diff * diff
    });
}

// Compare the distance to the point of the candidate point with
// the best candidate so far (whose distance is already computed).
// Return the closer point and the corresponding distance
fn closer_of<T: FriendlyFloat, const K: usize>(
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

fn sample_median<T: FriendlyFloat>(v: &mut Vec<T>, max_n_samples: Option<usize>) -> T {
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-8;

    // Run tests with a large number of points. This ensures that the
    // tests will take a noticeable long amount of time unless our
    // implementation is correct
    const N_POINTS: usize = 100000;

    // A random point from the [0, 1] K-hypercube
    fn random_point<T: FriendlyFloat, const K: usize>() -> [T; K] {
        let mut res = [0.0.into(); K];
        for i in 0..K {
            res[i] = rand::random::<f64>().into()
        }
        return res;
    }

    // A bunch of points on the [0, 1] K-hypercube
    fn fake_data<T: FriendlyFloat, const K: usize>() -> Vec<[T; K]> {
        return (0..N_POINTS)
            .map(|_| random_point())
            .collect::<Vec<[T; K]>>();
    }

    #[test]
    fn contains() {
        let points = fake_data();
        let tree = KDTree::new(&points);

        let p0 = points[0];
        // Kinda cool--nowhere above did we specify that K == 3, but the
        // compiler infers it from this declaration
        let p1 = [-0.3, 0.1, 0.7];
        assert!(tree.contains(&p0));
        assert!(!tree.contains(&p1));
    }

    #[test]
    fn nearest_neighbor() {
        let points = vec![
            [0.2, -0.49, 0.87, 0.89],
            [-1.3, 1.45, 1.41, 1.21],
            [1.29, -0.03, -0.18, 0.23],
            [0.79, 1.22, -0.76, -1.07],
            [-1.27, 1.22, 0.7, -0.69],
        ];
        let tree = KDTree::new(&points);

        let outsider = [0.2, -0.5, 0.9, 0.9];
        let expected_neighbor = points[0];
        let expected_distance = 0.0011;

        let (neighbor, distance) = tree.nearest_neighbor(&outsider, None);

        assert_eq!(expected_neighbor, neighbor);
        assert!((distance - expected_distance).abs() <= EPSILON);
    }

    // Private helper functions

    #[test]
    fn min_lr_() {
        // Left
        {
            let lr = min_lr(0.4, 0.5);
            assert!(lr.0 == 0.0);
            assert!((lr.1 - 0.1 * 0.1).abs() <= EPSILON);
        }
        // Right
        {
            let lr = min_lr(-0.4, -0.5);
            assert!((lr.0 - 0.1 * 0.1).abs() <= EPSILON);
            assert!(lr.1 == 0.0);
        }
        // Center
        {
            let lr = min_lr(0.5, 0.5);
            assert!(lr.0.abs() <= EPSILON);
            assert!(lr.1.abs() <= EPSILON);
        }
    }

    #[test]
    fn squared_distance_() {
        let point0 = [3.0, -0.5];
        let point1 = [1.0, 0.5];
        let expected = 5.0;
        assert!((squared_distance(&point0, &point1) - expected).abs() <= EPSILON);
    }

    #[test]
    fn closer_of_() {
        // Candidate is better than best_so_far
        {
            let point = [1.0, 2.0];
            let candidate = [2.0, 2.0];
            let best_so_far = ([2.0, 1.0], 2.0);
            // Check that our "precomputed" distance was correct :)
            assert!((squared_distance(&point, &best_so_far.0) - 2.0).abs() <= EPSILON);
            let closer = closer_of(&point, candidate, best_so_far);
            assert_eq!(candidate, closer.0);
            assert!((closer.1 - 1.0).abs() <= EPSILON);
        }

        // Candidate is worse than best_so_far
        {
            let point = [1.0, 2.0];
            let candidate = [2.0, 1.0];
            let best_so_far = ([2.0, 2.0], 1.0);
            // Check that our "precomputed" distance was correct :)
            assert!((squared_distance(&point, &best_so_far.0) - 1.0).abs() <= EPSILON);
            let closer = closer_of(&point, candidate, best_so_far);
            assert_eq!(best_so_far.0, closer.0);
            assert!((closer.1 - 1.0).abs() <= EPSILON);
        }
    }

    #[test]
    fn sample_median_() {
        // Even length
        {
            let mut v = vec![1.0, 2.0];
            assert!((sample_median(&mut v, None) - 1.5) <= EPSILON);
        }
        // Even length
        {
            let mut v = vec![1.0, 2.0, 3.0, 400.0];
            assert!((sample_median(&mut v, None) - 2.5) <= EPSILON);
        }
        // Odd length
        {
            let mut v = vec![3.14];
            assert_eq!(sample_median(&mut v, None), 3.14);
        }
        // Odd length
        {
            let mut v = vec![3.14, -1.0, 5.5, -16.7, 27.0];
            assert_eq!(sample_median(&mut v, None), 3.14);
        }
        // Empty: should error
        // {
        //     let mut v: Vec<f64> = vec![];
        //     assert_eq!(sample_median(&mut v, None), 3.14);
        // }
    }
}
