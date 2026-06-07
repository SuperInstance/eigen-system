//! Inverse iteration for finding eigenvalues close to a given value.

use crate::Matrix;

/// Inverse iteration to find eigenvalue closest to sigma.
pub fn inverse_iteration(a: &Matrix, sigma: f64, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = a.rows;
    let mut a_shifted = a.clone();
    for i in 0..n {
        a_shifted.set(i, i, a_shifted.get(i, i) - sigma);
    }

    let mut v = vec![1.0; n];
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for vi in &mut v {
        *vi /= norm;
    }

    let mut mu = sigma;
    for _ in 0..max_iter {
        let w = match a_shifted.solve(&v) {
            Some(w) => w,
            None => break,
        };

        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();

        // Rayleigh quotient for eigenvalue estimate
        let av = a.mul_vec(&v);
        let mu_new = v.iter().zip(av.iter()).map(|(vi, ai)| vi * ai).sum::<f64>();

        if (mu_new - mu).abs() < tol {
            return (mu_new, v);
        }
        mu = mu_new;
    }

    let av = a.mul_vec(&v);
    let eigenvalue = v.iter().zip(av.iter()).map(|(vi, ai)| vi * ai).sum::<f64>();
    (eigenvalue, v)
}

/// Find multiple eigenvalues using inverse iteration with different shifts.
pub fn find_eigenvalues(a: &Matrix, shifts: &[f64], max_iter: usize, tol: f64) -> Vec<(f64, Vec<f64>)> {
    shifts
        .iter()
        .map(|&sigma| inverse_iteration(a, sigma, max_iter, tol))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_3x3() -> Matrix {
        Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    }

    #[test]
    fn test_inverse_iteration_close_to_2() {
        let a = diag_3x3();
        let (eig, v) = inverse_iteration(&a, 2.1, 100, 1e-10);
        assert!((eig - 2.0).abs() < 1e-6, "got eigenvalue {}", eig);
        // Check eigenvector
        let av = a.mul_vec(&v);
        for i in 0..3 {
            assert!((av[i] - eig * v[i]).abs() < 1e-4, "eigenvector check failed at {}", i);
        }
    }

    #[test]
    fn test_inverse_iteration_close_to_1() {
        let a = diag_3x3();
        let (eig, _) = inverse_iteration(&a, 0.9, 100, 1e-10);
        assert!((eig - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_multiple_eigenvalues() {
        let a = diag_3x3();
        let results = find_eigenvalues(&a, &[0.5, 1.8, 3.1], 100, 1e-10);
        let eigs: Vec<f64> = results.iter().map(|(e, _)| *e).collect();
        assert!(eigs.iter().any(|e| (e - 1.0).abs() < 1e-4));
        assert!(eigs.iter().any(|e| (e - 2.0).abs() < 1e-4));
        assert!(eigs.iter().any(|e| (e - 3.0).abs() < 1e-4));
    }

    #[test]
    fn test_inverse_symmetric() {
        let a = Matrix::from_vec(2, 2, vec![1.5, 0.5, 0.5, 1.5]);
        let (eig, _) = inverse_iteration(&a, 1.9, 100, 1e-10);
        assert!((eig - 2.0).abs() < 1e-6);
    }
}
