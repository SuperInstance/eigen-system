//! Tridiagonal reduction using Householder reflections.

use crate::Matrix;

/// Reduce a symmetric matrix to tridiagonal form using Householder reflections.
///
/// Returns the tridiagonal matrix.
pub fn tridiagonalize(a: &Matrix) -> Matrix {
    let n = a.rows;
    let mut t = a.clone();

    for k in 0..n.saturating_sub(2) {
        // Extract subdiagonal column
        let mut x = vec![0.0; n - k - 1];
        for i in (k + 1)..n {
            x[i - k - 1] = t.get(i, k);
        }

        let norm_x = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        if norm_x < 1e-14 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;
        let norm_v = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        if norm_v < 1e-14 {
            continue;
        }
        for xi in &mut x {
            *xi /= norm_v;
        }

        // T = (I - 2vv^T) T (I - 2vv^T)
        // Apply from left: T = (I - 2vv^T) T
        for j in 0..n {
            let dot: f64 = (0..x.len()).map(|i| x[i] * t.get(k + 1 + i, j)).sum();
            for i in 0..x.len() {
                t.set(k + 1 + i, j, t.get(k + 1 + i, j) - 2.0 * x[i] * dot);
            }
        }

        // Apply from right: T = T (I - 2vv^T)
        for i in 0..n {
            let dot: f64 = (0..x.len()).map(|j2| x[j2] * t.get(i, k + 1 + j2)).sum();
            for j2 in 0..x.len() {
                t.set(i, k + 1 + j2, t.get(i, k + 1 + j2) - 2.0 * dot * x[j2]);
            }
        }
    }

    t
}

/// Check if a matrix is tridiagonal (within tolerance).
pub fn is_tridiagonal(a: &Matrix, tol: f64) -> bool {
    for i in 0..a.rows {
        for j in 0..a.cols {
            if (i as i64 - j as i64).unsigned_abs() > 1 && a.get(i, j).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Extract diagonal and sub/super-diagonal from a tridiagonal matrix.
pub fn extract_bands(a: &Matrix) -> (Vec<f64>, Vec<f64>) {
    let n = a.rows;
    let diag: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
    let subdiag: Vec<f64> = (0..n - 1).map(|i| a.get(i + 1, i)).collect();
    (diag, subdiag)
}

/// Eigenvalues of a symmetric tridiagonal matrix using the QR algorithm.
pub fn tridiag_eigenvalues(diag: &[f64], subdiag: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    let n = diag.len();
    let mut d = diag.to_vec();
    let mut e = vec![0.0; n];
    e[..subdiag.len()].copy_from_slice(subdiag);

    // Implicit QR with Wilkinson shift
    let mut m = n;
    for _ in 0..max_iter {
        if m <= 1 {
            break;
        }

        // Check for convergence of last subdiagonal element
        while m > 1 && e[m - 2].abs() < tol * (d[m - 1].abs() + d[m - 2].abs()) {
            m -= 1;
        }
        if m <= 1 {
            break;
        }

        // Wilkinson shift
        let dd = (d[m - 1] - d[m - 2]) / 2.0;
        let shift = d[m - 1] - e[m - 2] * e[m - 2] / (dd + dd.signum() * (dd * dd + e[m - 2] * e[m - 2]).sqrt());

        let mut x = d[0] - shift;
        let mut y = e[0];

        for k in 0..m - 1 {
            let r = (x * x + y * y).sqrt();
            let c = if r.abs() > 1e-30 { x / r } else { 1.0 };
            let s = if r.abs() > 1e-30 { y / r } else { 0.0 };

            let w = c * x + s * y;
            if k > 0 {
                e[k - 1] = w;
            }

            let q = c * d[k] + s * e[k];
            let z = -s * d[k] + c * e[k];
            let p = c * e[k] + s * d[k + 1];
            let r_val = -s * e[k] + c * d[k + 1];

            d[k] = c * q + s * p;
            e[k] = -s * q + c * p;
            d[k + 1] = -s * z + c * r_val;

            if k < m - 2 {
                x = e[k];
                y = s * e[k + 1];
                e[k + 1] = c * e[k + 1];
            }
        }
    }

    let mut eigenvalues = d;
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_3x3() -> Matrix {
        Matrix::from_vec(3, 3, vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0])
    }

    #[test]
    fn test_tridiagonalize() {
        let a = symmetric_3x3();
        let t = tridiagonalize(&a);
        assert!(is_tridiagonal(&t, 1e-10));
    }

    #[test]
    fn test_tridiagonalize_preserves_trace() {
        let a = symmetric_3x3();
        let t = tridiagonalize(&a);
        let trace_a: f64 = (0..3).map(|i| a.get(i, i)).sum();
        let trace_t: f64 = (0..3).map(|i| t.get(i, i)).sum();
        assert!((trace_a - trace_t).abs() < 1e-10);
    }

    #[test]
    fn test_is_tridiagonal() {
        let t = Matrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 2.0, 3.0, 4.0, 0.0, 4.0, 5.0]);
        assert!(is_tridiagonal(&t, 1e-10));
        let a = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(!is_tridiagonal(&a, 1e-10));
    }

    #[test]
    fn test_extract_bands() {
        let t = Matrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 2.0, 3.0, 4.0, 0.0, 4.0, 5.0]);
        let (diag, subdiag) = extract_bands(&t);
        assert_eq!(diag, vec![1.0, 3.0, 5.0]);
        assert_eq!(subdiag, vec![2.0, 4.0]);
    }

    #[test]
    fn test_tridiag_eigenvalues() {
        let diag = vec![2.0, 3.0, 5.0];
        let subdiag = vec![0.0, 0.0];
        let eigs = tridiag_eigenvalues(&diag, &subdiag, 100, 1e-10);
        assert!((eigs[0] - 5.0).abs() < 1e-6);
        assert!((eigs[1] - 3.0).abs() < 1e-6);
        assert!((eigs[2] - 2.0).abs() < 1e-6);
    }
}
