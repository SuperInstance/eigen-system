//! # eigen-system
//!
//! Eigenvalue computation: power iteration, QR algorithm, tridiagonal reduction, inverse iteration.
//!
//! ## Modules
//! - `power` — Power iteration and inverse power iteration
//! - `qr` — QR algorithm for eigenvalue computation
//! - `tridiag` — Tridiagonal reduction (Householder)
//! - `inverse` — Inverse iteration for specific eigenvalues
//! - `shift` — Shifted QR and Rayleigh quotient iteration

#![allow(unknown_lints, clippy::needless_range_loop, clippy::assign_op_pattern, clippy::identity_op, clippy::ptr_arg)]

pub mod power;
pub mod qr;
pub mod tridiag;
pub mod inverse;
pub mod shift;

/// A simple matrix type using row-major storage.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Matrix-vector multiplication.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols);
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0;
                for k in 0..self.cols {
                    s += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, s);
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Solve Ax = b using Gaussian elimination with partial pivoting.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        let n = self.rows;
        assert_eq!(n, self.cols);
        assert_eq!(n, b.len());

        let mut aug = vec![vec![0.0; n + 1]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = self.get(i, j);
            }
            aug[i][n] = b[i];
        }

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in col + 1..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return None;
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            for j in col..=n {
                aug[col][j] /= pivot;
            }
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for j in col..=n {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }

        Some((0..n).map(|i| aug[i][n]).collect())
    }
}
