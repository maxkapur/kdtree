# kdtree

A pretty Spartan implementation of a `k`-d tree in Rust.

Example:

```rust
use kdtree::KDTree;
let points = vec![
    [0.2, -0.49, 0.87, 0.89],
    [-1.3, 1.45, 1.41, 1.21],
    [1.29, -0.03, -0.18, 0.23],
    [0.79, 1.22, -0.76, -1.07],
    [-1.27, 1.22, 0.7, -0.69],
];
let tree = KDTree::new(&points);
assert!(tree.contains(&[0.2, -0.49, 0.87, 0.89]));
assert!(!tree.contains(&[-0.2, -0.49, 0.87, 0.89]));
```

Author: Max Kapur ([email](mailto:max@maxkapur.com))
