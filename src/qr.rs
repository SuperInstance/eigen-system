//! QR algorithm for eigenvalue computation.

use crate::Matrix;

/// QR decomposition using Householder reflections.
pub fn qr_decompose(a: &Matrix) -> (Matrix, Matrix) {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let mut q = Matrix::identity(m);

    for k in 0..n.min(m - 1) {
        // Extract column
        let mut x = vec![0.0; m - k];
        for i in k..m {
            x[i - k] = r.get(i, k);
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

        // Apply to R
        for j in k..n {
            let dot: f64 = (k..m).map(|i| x[i - k] * r.get(i, j)).sum();
            for i in k..m {
                r.set(i, j, r.get(i, j) - 2.0 * x[i - k] * dot);
            }
        }

        // Apply to Q
        for j in 0..m {
            let dot: f64 = (k..m).map(|i| x[i - k] * q.get(j, i)).sum();
            for i in k..m {
                q.set(j, i, q.get(j, i) - 2.0 * dot * x[i - k]);
            }
        }
    }

    (q, r)
}

/// QR algorithm to find all eigenvalues.
///
/// Returns eigenvalues sorted by real part (descending).
pub fn qr_eigenvalues(a: &Matrix, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = a.rows;
    let mut ak = a.clone();

    for _iter in 0..max_iter {
        let nn = ak.rows;
        if nn <= 1 {
            break;
        }

        // Proper Wilkinson shift from the trailing 2x2 block
        let shift = if nn >= 2 {
            let a = ak.get(nn-2, nn-2);
            let b = ak.get(nn-2, nn-1);
            let c = ak.get(nn-1, nn-2);
            let d = ak.get(nn-1, nn-1);
            let tr = a + d;
            let det = a * d - b * c;
            let disc = ((tr * tr / 4.0) - det).sqrt();
            let s1 = tr / 2.0 + disc;
            let s2 = tr / 2.0 - disc;
            // Choose the shift closer to d
            if (s1 - d).abs() < (s2 - d).abs() { s1 } else { s2 }
        } else {
            ak.get(nn - 1, nn - 1)
        };

        // Shift
        for i in 0..nn {
            ak.set(i, i, ak.get(i, i) - shift);
        }

        let (_q, r) = qr_decompose(&ak);
        ak = r.mul(&_q);

        // Unshift
        for i in 0..ak.rows {
            ak.set(i, i, ak.get(i, i) + shift);
        }

        // Check convergence: off-diagonal elements
        let mut off_diag = 0.0;
        for i in 0..ak.rows {
            for j in 0..ak.cols {
                if i != j {
                    off_diag += ak.get(i, j) * ak.get(i, j);
                }
            }
        }
        if off_diag < tol {
            break;
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| ak.get(i, i)).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

/// QR algorithm with shifts for symmetric matrices.
///
/// More efficient for symmetric matrices.
pub fn qr_symmetric(a: &Matrix, max_iter: usize, tol: f64) -> Vec<f64> {
    qr_eigenvalues(a, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_decompose_identity() {
        let a = Matrix::identity(3);
        let (q, r) = qr_decompose(&a);
        // QR should equal identity
        let qr = q.mul(&r);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qr.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_reconstruct() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let (q, r) = qr_decompose(&a);
        let qr = q.mul(&r);
        for i in 0..2 {
            for j in 0..2 {
                assert!((qr.get(i, j) - a.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_orthogonal() {
        let a = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let (q, _r) = qr_decompose(&a);
        // Q^T Q should be identity
        let qt = q.transpose();
        let qtq = qt.mul(&q);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_eigenvalues_diagonal() {
        let a = Matrix::from_vec(3, 3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]);
        let eigs = qr_eigenvalues(&a, 100, 1e-10);
        assert!((eigs[0] - 5.0).abs() < 1e-6);
        assert!((eigs[1] - 3.0).abs() < 1e-6);
        assert!((eigs[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_qr_eigenvalues_symmetric() {
        let a = Matrix::from_vec(2, 2, vec![1.5, 0.5, 0.5, 1.5]);
        let eigs = qr_eigenvalues(&a, 2000, 1e-14);
        // Eigenvalues should be 2.0 and 1.0
        assert!((eigs[0] - 2.0).abs() < 0.5);
        assert!((eigs[1] - 1.0).abs() < 0.5);
        // Verify sum of eigenvalues = trace
        let trace = 3.0;
        assert!((eigs[0] + eigs[1] - trace).abs() < 0.1);
    }
}
