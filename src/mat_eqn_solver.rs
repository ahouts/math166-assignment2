use crate::mat::{Mat, RowOperation};
use crate::upper_triangle::UpperTriangle;
use std::marker::PhantomData;

// trait represeting a solution to a matrix equation
pub trait MatEqnSolver {
    fn solve(m: Mat, b: Vec<f64>) -> Result<Vec<f64>, ()>;
}

// get the matrix into upper triangular form, then use
// reverse substitution to get the solution
pub struct ReverseSub<T: UpperTriangle> {
    t: PhantomData<*const T>,
}

impl<T> MatEqnSolver for ReverseSub<T>
where
    T: UpperTriangle,
{
    fn solve(mut m: Mat, mut b: Vec<f64>) -> Result<Vec<f64>, ()> {
        assert_eq!(m.cols(), b.len());
        let res = T::process(&mut m, &mut |op| match op {
            &RowOperation::Swap(r1, r2) => b.swap(r1, r2),
            &RowOperation::Scale { src, scale, dest } => b[dest] = scale * b[src],
        });
        if let Err(_) = res {
            return Err(());
        }
        let n = m.rows() - 1;
        let mut res = vec![0.0; m.rows()];
        res[n] = b[n] / m.get(n, n);
        for i in (0..=n).rev() {
            res[i] =
                (b[i] - ((i + 1)..=n).map(|j| m.get(i, j) * res[j]).sum::<f64>()) / m.get(i, i);
        }
        Ok(res)
    }
}
