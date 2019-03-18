use crate::mat::{Mat, RowOperation};
use crate::upper_triangle::UpperTriangle;
use std::marker::PhantomData;

// trait representing a method of LU decomposition
pub trait LuDec {
    fn dec(m: Mat) -> Result<(Mat, Mat), ()>;
}

// doolittle algorithm to calculate LU decomposition,
// utilizing given UpperTriangle formula
pub struct Doolittle<T: UpperTriangle> {
    t: PhantomData<*const T>,
}

impl<T: UpperTriangle> LuDec for Doolittle<T> {
    fn dec(mut u: Mat) -> Result<(Mat, Mat), ()> {
        assert!(u.is_square());
        let mut l = Mat::new_i(u.rows());
        // run the UpperTriangle formula,
        // executing the closure every time a row operation is performed
        T::run(&mut u, &mut |op| match *op {
            RowOperation::Cmb { src, scale, dest } => l.set(dest, src, scale),
            RowOperation::Swap(r1, r2) => eprintln!(
                "tried to swap R{} and R{}, but swapping is not supported.",
                r1, r2
            ),
            RowOperation::Scale { row, .. } => {
                eprintln!("tried to scale R{}, but scaling is not supported", row)
            }
        })?;
        Ok((l, u))
    }
}
