use crate::mat::Mat;
use crate::reduce_upper::ReduceUpper;
use crate::upper_triangle::UpperTriangle;
use std::marker::PhantomData;

// trait representing the process of inverting a matrix
pub trait Invert {
    fn invert(m: Mat) -> Result<Mat, ()>;
}

// create an identity matrix x
// invert matrix by performing row operations on the input matrix until
// it is the identity matrix with algorithms T and R
// performing the same operations on x
pub struct AugmentedMat<T: UpperTriangle, R: ReduceUpper> {
    t: PhantomData<*const T>,
    r: PhantomData<*const R>,
}

impl<T: UpperTriangle, R: ReduceUpper> Invert for AugmentedMat<T, R> {
    fn invert(mut m: Mat) -> Result<Mat, ()> {
        assert!(m.is_square());
        let mut res = Mat::new_i(m.rows());
        T::run(&mut m, &mut |op| {
            res.apply(op);
        })?;
        R::run(&mut m, &mut |op| {
            res.apply(op);
        })?;
        Ok(res)
    }
}
