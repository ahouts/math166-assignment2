use crate::mat::{Mat, RowOperation};
use crate::upper_triangle::UpperTriangle;
use std::marker::PhantomData;

pub trait LuDec {
    fn dec(m: Mat) -> Result<(Mat, Mat), ()>;
}

pub struct Doolittle<T: UpperTriangle> {
    t: PhantomData<*const T>,
}

impl<T: UpperTriangle> LuDec for Doolittle<T> {
    fn dec(mut u: Mat) -> Result<(Mat, Mat), ()> {
        assert!(u.is_square());
        let mut l = Mat::new_i(u.rows());
        T::run(&mut u, &mut |op| match *op {
            RowOperation::Cmb { src, scale, dest } => l.set(dest, src, scale),
            RowOperation::Swap(r1, r2) => eprintln!(
                "tried to swap R{} and R{}, but swapping is not supported.",
                r1, r2
            ),
            _ => {}
        })?;
        Ok((l, u))
    }
}
