mod invert;
mod lu_dec;
mod mat;
mod mat_eqn_solver;
mod reduce_upper;
mod upper_triangle;

use crate::invert::{AugmentedMat, Invert};
use crate::reduce_upper::BasicReduceUpper;
use crate::upper_triangle::Gaussian;

fn main() {
    let mut m = crate::mat::Mat::new(3, 3);
    m.set(0, 0, 1.0);
    m.set(0, 1, 1.0);
    m.set(0, 2, 1.0);

    m.set(1, 0, 0.0);
    m.set(1, 1, 2.0);
    m.set(1, 2, 5.0);

    m.set(2, 0, 2.0);
    m.set(2, 1, 5.0);
    m.set(2, 2, -1.0);

    println!("{}", m);
    let mp = AugmentedMat::<Gaussian, BasicReduceUpper>::invert(m).expect("err");
    println!("{}", mp);
}
