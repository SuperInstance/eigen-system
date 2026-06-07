# eigen-system

Research-grade eigenvalue computation in pure Rust.

## Features

- **Power iteration**: Dominant eigenvalue/eigenvector, inverse power iteration
- **QR algorithm**: QR decomposition, QR eigenvalue computation with shifts
- **Tridiagonal reduction**: Householder tridiagonalization, banded eigenvalue solver
- **Inverse iteration**: Find eigenvalues near a given shift
- **Shift**: Rayleigh quotient iteration, shifted QR

## Usage

```rust
use eigen_system::power::power_iteration;
use eigen_system::Matrix;

fn main() {
    let a = Matrix::from_vec(2, 2, vec![1.5, 0.5, 0.5, 1.5]);
    let result = power_iteration(&a, 1000, 1e-10);
    println!("Dominant eigenvalue: {:.6}", result.eigenvalue);
}
```

## No Dependencies

This crate uses only `std`.

## License

MIT OR Apache-2.0
