use num::Float;
use rand::prelude::*;
use std::rc::Rc;

// When calculating the median, sample (randomly) at least this
// many points to reduce time required to build the tree.
const MAX_N_SAMPLES: usize = 100;

/// A FriendlyFloat is a trait alias for the combination of traits
/// needed to create a KDTree with elements of type T: FriendlyFloat.
pub trait FriendlyFloat: PartialOrd + Float + From<f64> {}
impl<T: PartialOrd + Float + From<f64>> FriendlyFloat for T {}

/// KDTree is an enum which could be either a Stem or Leaf.
/// Calling KDTree::new(&points) will (typically) produce a
/// Stem, which branches out to define the whole tree, whose
/// leaves are Leaf instances which store the point data.
pub enum KDTree<T: FriendlyFloat, const K: usize> {
    Stem(Rc<Stem<T, K>>),
    Leaf(Rc<Leaf<T, K>>),
}

impl<T: FriendlyFloat, const K: usize> KDTree<T, K> {
    /// Create a new KDTree containing the points provided.
    ///
    /// ```
    /// use kdtree::KDTree;
    /// let points = vec![
    ///     [0.2, -0.49, 0.87, 0.89],
    ///     [-1.3, 1.45, 1.41, 1.21],
    ///     [1.29, -0.03, -0.18, 0.23],
    ///     [0.79, 1.22, -0.76, -1.07],
    ///     [-1.27, 1.22, 0.7, -0.69],
    /// ];
    /// let tree = KDTree::new(&points);
    /// ```
    pub fn new(points: &Vec<[T; K]>) -> KDTree<T, K> {
        return KDTree::new_with_axis(&points, None);
    }

    // Recursive KDTree constructor. Do a median split along axis k,
    // and construct a Stem node with subtrees at depth k + 1 mod K.
    // If points has only one element, construct a Leaf node instead.
    fn new_with_axis(points: &Vec<[T; K]>, k: Option<usize>) -> KDTree<T, K> {
        if points.len() == 1 {
            let point = points[0];
            return KDTree::Leaf(Leaf { point }.into());
        }
        let k = k.unwrap_or(0);
        let median = {
            let mut v = points.iter().map(|point| point[k]).collect::<Vec<T>>();
            sample_median(&mut v, None)
        };
        let (left, right) = {
            let mut left_points = Vec::<[T; K]>::with_capacity(points.len() / 2);
            let mut right_points = Vec::<[T; K]>::with_capacity(points.len() / 2);
            for point in points {
                if point[k] <= median {
                    left_points.push(*point);
                } else {
                    right_points.push(*point);
                }
            }
            let left = KDTree::new_with_axis(&left_points, Some((k + 1) % K));
            let right = KDTree::new_with_axis(&right_points, Some((k + 1) % K));
            (left, right)
        };
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

    /// True if the tree contains the point provided.
    ///
    /// ```
    /// use kdtree::KDTree;
    /// let points = vec![
    ///     [0.2, -0.49, 0.87, 0.89],
    ///     [-1.3, 1.45, 1.41, 1.21],
    ///     [1.29, -0.03, -0.18, 0.23],
    ///     [0.79, 1.22, -0.76, -1.07],
    ///     [-1.27, 1.22, 0.7, -0.69],
    /// ];
    /// let tree = KDTree::new(&points);
    /// assert!(tree.contains(&[0.2, -0.49, 0.87, 0.89]));
    /// assert!(!tree.contains(&[-0.2, -0.49, 0.87, 0.89]));
    /// ```
    pub fn contains(&self, point: &[T; K]) -> bool {
        return match self {
            KDTree::Leaf(leaf) => leaf.matches(point),
            KDTree::Stem(stem) => stem.contains(point),
        };
    }

    /// The nearest point in the tree to that provided, and the
    /// squared distance to it.
    ///
    /// ```
    /// use kdtree::KDTree;
    /// let points = vec![
    ///     [0.2, -0.49, 0.87, 0.89],
    ///     [-1.3, 1.45, 1.41, 1.21],
    ///     [1.29, -0.03, -0.18, 0.23],
    ///     [0.79, 1.22, -0.76, -1.07],
    ///     [-1.27, 1.22, 0.7, -0.69],
    /// ];
    /// let tree = KDTree::new(&points);
    /// let outsider = [0.2, -0.5, 0.9, 0.9];
    /// let expected_neighbor = points[0];
    /// let (neighbor, _distance) = tree.nearest_neighbor(&outsider);
    /// assert_eq!(expected_neighbor, neighbor);
    /// ```
    pub fn nearest_neighbor(&self, point: &[T; K]) -> ([T; K], T) {
        let mut incumbent = Incumbent::dummy();
        self.nearest_neighbor_with_incumbent(point, &mut incumbent);
        return (incumbent.point, incumbent.distance);
    }

    /// The nearest point in the tree to that provided, but skip
    /// exploring nodes for which the distance to the point cannot be
    /// better than the incumbent. Iteratively update incumbent in place
    /// as we explore.
    fn nearest_neighbor_with_incumbent(&self, point: &[T; K], incumbent: &mut Incumbent<T, K>) {
        match self {
            KDTree::Leaf(leaf) => leaf.update(point, incumbent),
            KDTree::Stem(stem) => stem.dfs_nearest_neighbor(point, incumbent),
        }
    }
}

/// A Leaf node in a KDTree is a single point in K-space.
pub struct Leaf<T: FriendlyFloat, const K: usize> {
    point: [T; K],
}

impl<T: FriendlyFloat, const K: usize> Leaf<T, K> {
    // True if the point provided equals, in each coordinate, the point
    // associated with this Leaf.
    fn matches(&self, point: &[T; K]) -> bool {
        return &self.point == point;
    }

    // Compare the distance between (the point represented by) this Leaf
    // and the target point, with the distance associated with the incumbent.
    // If this Leaf is closer, update the incumbent in place.
    fn update(&self, point: &[T; K], incumbent: &mut Incumbent<T, K>) {
        let my_distance = self.squared_distance(point);
        if my_distance < incumbent.distance {
            incumbent.point = self.point;
            incumbent.distance = my_distance;
        };
    }

    fn squared_distance(&self, point: &[T; K]) -> T {
        return squared_distance(&self.point, point);
    }
}

/// A Stem node in a KDTree defines a median and axis K on which the
/// tree splits into subtrees.
pub struct Stem<T: FriendlyFloat, const K: usize> {
    k: usize,
    median: T,
    left: KDTree<T, K>,
    right: KDTree<T, K>,
}

impl<T: FriendlyFloat, const K: usize> Stem<T, K> {
    // Use depth-first search to determine whether the point exists
    // among this Stem's arms.
    fn contains(&self, point: &[T; K]) -> bool {
        let left_side = point[self.k] <= self.median;
        if left_side {
            return self.left.contains(&point);
        } else {
            return self.right.contains(&point);
        };
    }

    // Use depth-first search to find the closest point among this Stem's
    // arms to the target point. Update the incumbent solution in place.
    fn dfs_nearest_neighbor(&self, point: &[T; K], incumbent: &mut Incumbent<T, K>) {
        // Any points in my left, right arm have distance at least
        let (min_left, min_right) = min_lr(point[self.k], self.median);
        // and we use this bound to skip exloring arms which cannot
        // contain a better solution than the incumbent.
        if min_left < incumbent.distance {
            self.left.nearest_neighbor_with_incumbent(&point, incumbent)
        }
        if min_right < incumbent.distance {
            self.right
                .nearest_neighbor_with_incumbent(&point, incumbent)
        }
    }
}

// Compute the sample median of the given vector. Use at most
// max_n_samples to decrease computation time. Needs a mutable borrow
// because we construct the sample by shuffling the vector in place.
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

/// The minimum squared distance achievable in the left and right
/// arms of a node when the current value and median are as given.
fn min_lr<T: FriendlyFloat>(value: T, median: T) -> (T, T) {
    let mut diff = median - value;
    diff = diff * diff;
    if value <= median {
        return (0.0.into(), diff);
    } else {
        return (diff, 0.0.into());
    };
}

// Square of the Euclidean distance between two points.
fn squared_distance<T: FriendlyFloat, const K: usize>(point0: &[T; K], point1: &[T; K]) -> T {
    let init: T = 0.0.into();
    return (0..K).fold(init, |accum, i| {
        let diff = point0[i] - point1[i];
        accum + diff * diff
    });
}

// Holds the incumbent closest point and its distance.
struct Incumbent<T: FriendlyFloat, const K: usize> {
    point: [T; K],
    distance: T,
}

impl<T: FriendlyFloat, const K: usize> Incumbent<T, K> {
    // A dummy incumbent with nan coordinates and infinite distance
    // for initializing the search.
    fn dummy() -> Incumbent<T, K> {
        let point = [f64::nan().into(); K];
        let distance = f64::infinity().into();
        return Incumbent { point, distance };
    }
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
        let (neighbor, distance) = tree.nearest_neighbor(&outsider);
        let expected_neighbor = points[0];
        assert_eq!(expected_neighbor, neighbor);
        let expected_distance = 0.0011;
        assert!((distance - expected_distance).abs() <= EPSILON);
    }

    // Private helper functions

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
    fn incumbent_update() {
        // Candidate is better than incumbent
        {
            let point = [1.0, 2.0];
            let candidate = Leaf { point: [2.0, 2.0] };

            let mut incumbent = Incumbent::dummy();
            incumbent.point = [2.0, 1.0];
            incumbent.distance = 2.0;
            // Check that our "precomputed" distance was correct :)
            assert!((squared_distance(&point, &incumbent.point) - 2.0).abs() <= EPSILON);

            candidate.update(&point, &mut incumbent);
            assert_eq!([2.0, 2.0], incumbent.point);
            assert!((incumbent.distance - 1.0).abs() <= EPSILON);
        }
        // Candidate is worse than incumbent
        {
            let point = [1.0, 2.0];
            let candidate = Leaf { point: [2.0, 1.0] };
            let mut incumbent = Incumbent::dummy();
            incumbent.point = [2.0, 2.0];
            incumbent.distance = 1.0;
            // Check that our "precomputed" distance was correct :)
            assert!((squared_distance(&point, &incumbent.point) - 1.0).abs() <= EPSILON);

            candidate.update(&point, &mut incumbent);
            assert_eq!([2.0, 2.0], incumbent.point);
            assert!((incumbent.distance - 1.0).abs() <= EPSILON);
        };
    }
}
