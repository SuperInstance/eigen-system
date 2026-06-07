//! Shifted QR and Rayleigh quotient iteration.

use crate::Matrix;

/// Rayleigh quotient iteration for fast eigenvalue convergence.
pub fn rayleigh_iteration(a: &Matrix, v0: &[f64], max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = a.rows;
    let mut v = v0.to_vec();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for vi in &mut v {
        *vi /= norm;
    }

    let av = a.mul_vec(&v);
    let mut mu: f64 = v.iter().zip(av.iter()).map(|(vi, ai)| vi * ai).sum();

    for _ in 0..max_iter {
        // Solve (A - mu*I) w = v
        let mut a_shifted = a.clone();
        for i in 0..n {
            a_shifted.set(i, i, a_shifted.get(i, i) - mu);
        }
        let w = match a_shifted.solve(&v) {
            Some(w) => w,
            None => break,
        };

        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();

        let av = a.mul_vec(&v);
        let mu_new: f64 = v.iter().zip(av.iter()).map(|(vi, ai)| vi * ai).sum();

        if (mu_new - mu).abs() < tol {
            return (mu_new, v);
        }
        mu = mu_new;
    }

    (mu, v)
}

/// Shifted QR iteration with explicit shifts.
pub fn shifted_qr(a: &Matrix, shifts: &[f64], max_iter_per_shift: usize, tol: f64) -> Vec<f64> {
    let n = a.rows;
    let mut ak = a.clone();

    for &shift in shifts {
        for _ in 0..max_iter_per_shift {
            // Apply shift
            for i in 0..n {
                ak.set(i, i, ak.get(i, i) - shift);
            }

            // QR decomposition
            let (q, r) = qr_factor(&ak);

            // A = R Q + shift*I
            ak = r.mul(&q);
            for i in 0..n {
                ak.set(i, i, ak.get(i, i) + shift);
            }

            // Check convergence
            let off = off_diagonal_norm(&ak);
            if off < tol {
                break;
            }
        }
    }

    // Additional iterations without shifts
    for _ in 0..max_iter_per_shift * 10 {
        let (q, r) = qr_factor(&ak);
        ak = r.mul(&q);
        if off_diagonal_norm(&ak) < tol {
            break;
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| ak.get(i, i)).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

fn qr_factor(a: &Matrix) -> (Matrix, Matrix) {
    let n = a.rows;
    let mut r = a.clone();
    let mut q = Matrix::identity(n);

    for k in 0..n.saturating_sub(1) {
        let mut x = vec![0.0; n - k];
        for i in k..n {
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

        for j in k..n {
            let dot: f64 = (k..n).map(|i| x[i - k] * r.get(i, j)).sum();
            for i in k..n {
                r.set(i, j, r.get(i, j) - 2.0 * x[i - k] * dot);
            }
        }

        for j in 0..n {
            let dot: f64 = (k..n).map(|i| x[i - k] * q.get(j, i)).sum();
            for i in k..n {
                q.set(j, i, q.get(j, i) - 2.0 * dot * x[i - k]);
            }
        }
    }

    (q, r)
}

fn off_diagonal_norm(a: &Matrix) -> f64 {
    let mut s = 0.0;
    for i in 0..a.rows {
        for j in 0..a.cols {
            if i != j {
                s += a.get(i, j) * a.get(i, j);
            }
        }
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_2x2() -> Matrix {
        Matrix::from_vec(2, 2, vec![1.5, 0.5, 0.5, 1.5])
    }

    #[test]
    fn test_rayleigh_iteration() {
        let a = symmetric_2x2();
        let v0 = vec![1.0, 0.5];
        let (eig, _) = rayleigh_iteration(&a, &v0, 50, 1e-12);
        assert!((eig - 2.0).abs() < 1e-8 || (eig - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_shifted_qr() {
        let a = Matrix::from_vec(3, 3, vec![2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0]);
        let eigs = shifted_qr(&a, &[4.0, 2.5], 20, 1e-10);
        assert!((eigs[0] - 5.0).abs() < 1e-6);
        assert!((eigs[1] - 3.0).abs() < 1e-6);
        assert!((eigs[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rayleigh_convergence() {
        let a = symmetric_2x2();
        let v0 = vec![0.7, 0.7];
        let (eig, _) = rayleigh_iteration(&a, &v0, 100, 1e-14);
        // Should converge to one of the eigenvalues
        assert!((eig - 2.0).abs() < 1e-8 || (eig - 1.0).abs() < 1e-8);
    }
}
