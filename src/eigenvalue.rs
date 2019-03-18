use crate::mat::Mat;
use std::cmp::Ordering;
use std::mem::swap;
use rand::random;
use crate::mat_eqn_solver::MatEqnSolver;
use std::marker::PhantomData;

pub trait EigenSolve {
    fn eigen_solve(mat: &Mat, q: f64, accuracy: f64) -> Result<f64, ()>;
}

pub struct PowerMethod;

impl EigenSolve for PowerMethod {
    fn eigen_solve(m: &Mat, _: f64, accuracy: f64) -> Result<f64, ()> {
        const MAX_ITER: usize = 10_000;

        assert!(m.is_square());
        let mut x: Vec<f64> = (0..m.rows()).map(|_| random::<f64>() * 2. - 1.).collect();
        let mut u_prev = 0.;
        let mut curr_iter = 0;
        loop {
            if curr_iter >= MAX_ITER {
                return Err(());
            }
            let xpk_index = find_max_mag(&x);
            let xpk = x[xpk_index];

            let y = m * &x;
            let u = y[xpk_index];

            if f64::abs(u - u_prev) < accuracy {
                return Ok(u);
            }

            let ypk_index = find_max_mag(&y);
            let ypk = y[ypk_index];

            let mut x1: Vec<f64> = y.iter().map(|v| *v / ypk).collect();
            swap(&mut x, &mut x1);
            u_prev = u;
            curr_iter += 1;
        }
    }
}

fn find_max_mag<T: AsRef<[f64]>>(v: T) -> usize {
    let x = v.as_ref();
    let i = (0..x.len())
        .max_by(|v1, v2| {
            let max = f64::max(x[*v1].abs(), x[*v2].abs());
            if max == x[*v1].abs() {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        })
        .unwrap_or(0);
    i
}

pub struct InversePowerMethod<S: MatEqnSolver> {
    s: PhantomData<*const S>,
}

impl<S: MatEqnSolver> EigenSolve for InversePowerMethod<S> {
    fn eigen_solve(mat: &Mat, q: f64, accuracy: f64) -> Result<f64, ()> {
        const MAX_ITER: usize = 10_000;

        assert!(mat.is_square());
        let m = mat - &(q * &Mat::new_i(mat.rows()));

        let mut x: Vec<f64> = (0..m.rows()).map(|_| random::<f64>() * 2. - 1.).collect();
        let mut u_prev = 0.;
        let mut curr_iter = 0;
        loop {
            if curr_iter >= MAX_ITER {
                return Err(());
            }
            let xpk_index = find_max_mag(&x);
            let xpk = x[xpk_index];

            let y = S::solve(m.clone(), x.clone())?;
            let u = 1. / y[xpk_index] + q;

            if f64::abs(u - u_prev) < accuracy {
                return Ok(u);
            }

            let ypk_index = find_max_mag(&y);
            let ypk = y[ypk_index];

            let mut x1: Vec<f64> = y.iter().map(|v| *v / ypk).collect();
            swap(&mut x, &mut x1);
            u_prev = u;
            curr_iter += 1;
        }
    }
}
