//! Power iteration and inverse power iteration.

use crate::Matrix;

/// Result of power iteration.
pub struct PowerResult {
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub iterations: usize,
}

/// Power iteration to find the dominant eigenvalue and eigenvector.
pub fn power_iteration(a: &Matrix, max_iter: usize, tol: f64) -> PowerResult {
    let n = a.rows;
    let mut v = vec![1.0; n];
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for vi in &mut v {
        *vi /= norm;
    }

    let mut lambda = 0.0;
    for iter in 0..max_iter {
        let w = a.mul_vec(&v);
        let lambda_new = w.iter().zip(v.iter()).map(|(wi, vi)| wi * vi).sum::<f64>();
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();

        if (lambda_new - lambda).abs() < tol {
            return PowerResult {
                eigenvalue: lambda_new,
                eigenvector: v,
                iterations: iter + 1,
            };
        }
        lambda = lambda_new;
    }

    PowerResult {
        eigenvalue: lambda,
        eigenvector: v,
        iterations: max_iter,
    }
}

/// Inverse power iteration to find eigenvalue closest to a given shift.
pub fn inverse_power_iteration(a: &Matrix, shift: f64, max_iter: usize, tol: f64) -> PowerResult {
    let n = a.rows;
    let mut a_shifted = a.clone();
    for i in 0..n {
        a_shifted.set(i, i, a_shifted.get(i, i) - shift);
    }

    let mut v = vec![1.0; n];
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for vi in &mut v {
        *vi /= norm;
    }

    let mut lambda = 0.0;
    for iter in 0..max_iter {
        let w = match a_shifted.solve(&v) {
            Some(w) => w,
            None => break,
        };
        let lambda_new = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum::<f64>();
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();

        let eigenvalue = shift + 1.0 / lambda_new;
        if (eigenvalue - lambda).abs() < tol {
            return PowerResult {
                eigenvalue,
                eigenvector: v,
                iterations: iter + 1,
            };
        }
        lambda = eigenvalue;
    }

    PowerResult {
        eigenvalue: lambda,
        eigenvector: v,
        iterations: max_iter,
    }
}

/// Compute Rayleigh quotient: v^T A v / v^T v.
pub fn rayleigh_quotient(a: &Matrix, v: &[f64]) -> f64 {
    let av = a.mul_vec(v);
    let num: f64 = v.iter().zip(av.iter()).map(|(vi, ai)| vi * ai).sum();
    let den: f64 = v.iter().map(|vi| vi * vi).sum();
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_2x2() -> Matrix {
        // eigenvalues: 4, -1
        Matrix::from_vec(2, 2, vec![1.5, 0.5, 0.5, 1.5])
    }

    fn symmetric_3x3() -> Matrix {
        // eigenvalues: 2, 3, 5 (diagonal)
        Matrix::from_vec(3, 3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0])
    }

    #[test]
    fn test_power_dominant() {
        let a = symmetric_2x2();
        let result = power_iteration(&a, 1000, 1e-10);
        assert!((result.eigenvalue - 2.0).abs() < 1e-6, "got {}", result.eigenvalue);
    }

    #[test]
    fn test_power_3x3_diagonal() {
        let a = symmetric_3x3();
        let result = power_iteration(&a, 1000, 1e-10);
        assert!((result.eigenvalue - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_inverse_power() {
        let a = symmetric_3x3();
        let result = inverse_power_iteration(&a, 3.1, 1000, 1e-10);
        assert!((result.eigenvalue - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_rayleigh_quotient() {
        let a = symmetric_3x3();
        let v = vec![0.0, 1.0, 0.0];
        let rq = rayleigh_quotient(&a, &v);
        assert!((rq - 3.0).abs() < 1e-10);
    }
}
