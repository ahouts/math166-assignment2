mod eigenvalue;
mod invert;
mod lu_dec;
mod mat;
mod mat_eqn_solver;
mod norm;
mod reduce_upper;
mod upper_triangle;

use crate::invert::{AugmentedMat, Invert};
use crate::lu_dec::{Doolittle, LuDec};
use crate::mat::Mat;
use crate::norm::LInf;
use crate::reduce_upper::BasicReduceUpper;
use crate::upper_triangle::Gaussian;
use crate::mat_eqn_solver::LuDecompSolver;
use rand::random;
use crate::eigenvalue::{EigenSolve, PowerMethod, InversePowerMethod};

fn to_mat<A: AsRef<[f64]>, R: AsRef<[A]>>(v2: R) -> Mat {
    let rows = v2.as_ref().len();
    let cols = v2.as_ref()[0].as_ref().len();
    let mut res = Mat::new(rows, cols);

    let rows = v2.as_ref();
    for (r, row) in v2.as_ref().iter().enumerate() {
        for (c, val) in row.as_ref().iter().enumerate() {
            res.set(r, c, *val);
        }
    }

    res
}

fn main() {
    let mut m = Mat::new(3, 3);
    m.set(0, 0, 1.0);
    m.set(0, 1, 1.0);
    m.set(0, 2, 1.0);

    m.set(1, 0, 0.0);
    m.set(1, 1, 2.0);
    m.set(1, 2, 5.0);

    m.set(2, 0, 2.0);
    m.set(2, 1, 5.0);
    m.set(2, 2, -1.0);

    println!("#############################################");
    println!("# Problem #1                                #");
    println!("#############################################");
    println!();

    let (l, u) = Doolittle::<Gaussian>::dec(m.clone()).expect("error while reducing matrix");
    println!("starting matrix:");
    println!("{}", m);
    println!("L");
    println!("{}", l);
    println!("U");
    println!("{}", u);

    println!();
    println!("#############################################");
    println!("# Problem #2                                #");
    println!("#############################################");
    println!();

    println!("starting matrix:");
    println!("{}", m);
    let mp = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(m.clone())
        .expect("error while inverting matrix");
    println!("inverse matrix:");
    println!("{}", mp);
    println!("A * A^-1");
    println!("{}", &m * &mp);

    println!();
    println!("#############################################");
    println!("# Problem #3                                #");
    println!("#############################################");
    println!();

    for k in 2..=6 {
        println!("Hilbert Matrix ({})", k);
        let h = Mat::new_hilbert(k);
        println!("{}", h);

        println!("Hilbert Matrix ({})-1", k);
        let h_inv = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(h.clone())
            .expect("error computing inverse of hilbert matrix");
        println!("{}", h_inv);

        let cond = h.k::<LInf>(&h_inv);
        println!("K(h) = {}", cond);

        let (l, u) =
            Doolittle::<Gaussian>::dec(h.clone()).expect("error while computing LU decomposition");
        println!("L");
        println!("{}", l);

        println!("U");
        println!("{}", u);

        let l_inv = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(l.clone())
            .expect("error computing inverse");
        let u_inv = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(u.clone())
            .expect("error computing inverse");

        let u_inv_l_inv = &u_inv * &l_inv;
        println!("Hilbert Matrix ({})-1 (U^-1 * L^-1)", k);
        println!("{}", u_inv_l_inv);

        let diff = &u_inv_l_inv - &h_inv;
        println!(
            "maximum difference in a term of inverse matrices = {}",
            diff.max()
        );

        println!();
    }

    println!();
    println!("#############################################");
    println!("# Problem #4                                #");
    println!("#############################################");
    println!();

    let m = {
        const MAT_DATA: [[f64; 4]; 4] = [
            [1., 0., 0., 0.],
            [4., 1., 0., 0.],
            [1., 3., 1., 0.],
            [6., 2., 9., 1.],
        ];
        to_mat(MAT_DATA)
    };

    println!("matrix:");
    println!("{}", m);

    let m_inv = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(m.clone())
        .expect("error computing inverse");
    println!("inverse:");
    println!("{}", m_inv);

    let m = {
        const MAT_DATA: [[f64; 5]; 5] = [
            [1., 0., 0., 0., 0.],
            [3., 1., 0., 0., 0.],
            [5., 2., 1., 0., 0.],
            [1., 5., 3., 1., 0.],
            [3., 2., 1., 2., 1.],
        ];
        to_mat(MAT_DATA)
    };

    println!("matrix:");
    println!("{}", m);

    let m_inv = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(m.clone())
        .expect("error computing inverse");
    println!("inverse:");
    println!("{}", m_inv);

    println!();
    println!("#############################################");
    println!("# Problem #5                                #");
    println!("#############################################");
    println!();

    // for matrices of size 2, 4, 8, etc...
    for size in 1..=7 {
        let size = 2f64.powi(size) as usize;
        let mut m = Mat::new_i(size);
        for i in 0..size {
            for j in 0..i {
                m.set(i, j, random::<f64>() * 2. - 1.);
            }
        }

        let m_inv = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(m.clone())
            .expect("error computing inverse");

        println!(
            "for lower triangular matrix of size {}x{}, entries in (-1, 1)",
            size, size
        );
        let k = m.k::<LInf>(&m_inv);
        println!("k = {}", k);
        println!("k^(1/{}) = {}", size, k.powf(1. / (size as f64)));
        println!();
    }

    println!();
    println!("#############################################");
    println!("# Problem #6                                #");
    println!("#############################################");
    println!();

    println!("Matrix (a)");
    let ma = {
        const MAT_DATA: [[f64; 3]; 3] = [
            [2., 1., 1.],
            [1., 2., 1.],
            [1., 1., 2.],
        ];
        to_mat(MAT_DATA)
    };
    println!("{}", ma);
    let e = PowerMethod::eigen_solve(&ma, 0., 10e-10)
        .expect("error computing eigenvalue using the power method");
    println!("largest eigenvalue: {:.10}", e);
    println!();


    println!("Matrix (b)");
    let mb = {
        const MAT_DATA: [[f64; 4]; 4] = [
            [1., 1., 0., 0.],
            [1., 2., 0., 1.],
            [0., 0., 3., 3.],
            [0., 1., 3., 2.],
        ];
        to_mat(MAT_DATA)
    };
    println!("{}", mb);
    let e = PowerMethod::eigen_solve(&mb, 0., 10e-10)
        .expect("error computing eigenvalue using the power method");
    println!("largest eigenvalue: {:.10}", e);
    println!();

    println!("Matrix (c)");
    let mc = {
        const MAT_DATA: [[f64; 4]; 4] = [
            [5., -2., -1./2., 3./2.],
            [-2., 5., 3./2., -1./2.],
            [-1./2., 3./2., 5., -2.],
            [3./2., -1./2., -2., 5.],
        ];
        to_mat(MAT_DATA)
    };
    println!("{}", mc);
    let e = PowerMethod::eigen_solve(&mc, 0., 10e-10)
        .expect("error computing eigenvalue using the power method");
    println!("largest eigenvalue: {:.10}", e);
    println!();

    println!("Matrix (d)");
    let md = {
        const MAT_DATA: [[f64; 4]; 4] = [
            [-4., 0., 1./2., 1./2.],
            [1./2., -2., 0., 1./2.],
            [1./2., 1./2., 0., 0.],
            [0., 1., 1., 4.],
        ];
        to_mat(MAT_DATA)
    };
    println!("{}", md);
    let e = PowerMethod::eigen_solve(&md, 0., 10e-10)
        .expect("error computing eigenvalue using the power method");
    println!("largest eigenvalue: {:.10}", e);
    println!();

    println!("because of Gershgorin's disk theorem, we know all 4 of the eigenvalues of m");
    println!("lie in the circle in the complex plan: center (5, 0), radius 4");
    println!();

    println!("checking every .13 in (1, 9) for eigenvalues");
    let mut results = vec![];
    let mut q = 1.;
    while q < 9. {
        let e = InversePowerMethod::<LuDecompSolver<Doolittle<Gaussian>>>::eigen_solve(&mc, q, 10e-10)
            .expect("error computing eigenvalue");

        if let None = results.iter().filter(|v| f64::abs(*v - e) < 0.1).next() {
            results.push(e);
        }
        q += 0.13;
    }

    println!("found eigenvalues: {:?}", results);
}
