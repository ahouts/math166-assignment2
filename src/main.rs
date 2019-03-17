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
use rand::random;

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
        let mut m = Mat::new(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                m.set(r, c, MAT_DATA[r][c]);
            }
        }
        m
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
        let mut m = Mat::new(5, 5);
        for r in 0..5 {
            for c in 0..5 {
                m.set(r, c, MAT_DATA[r][c]);
            }
        }
        m
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
}
