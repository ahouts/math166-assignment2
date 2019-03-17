use crate::lu_dec::Doolittle;
use crate::mat_eqn_solver::{LuDecompSolver, MatEqnSolver};
use crate::norm::Norm;
use crate::upper_triangle::Gaussian;
use std::fmt;
use std::ops::{Mul, Sub};

#[derive(Debug, Clone)]
pub enum RowOperation {
    Swap(usize, usize),
    Cmb { src: usize, scale: f64, dest: usize },
    Scale { row: usize, scale: f64 },
}

#[derive(Debug, Clone)]
pub struct Mat {
    cols: usize,
    data: Box<[f64]>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0);
        assert!(cols > 0);
        Mat {
            cols,
            data: vec![0.0; rows * cols].into_boxed_slice(),
        }
    }

    pub fn new_i(size: usize) -> Self {
        let mut res = Mat::new(size, size);
        let n = size - 1;
        for i in 0..=n {
            res.set(i, i, 1.0);
        }
        res
    }

    pub fn new_hilbert(size: usize) -> Self {
        let mut res = Mat::new(size, size);
        let n = size - 1;
        for i in 0..=n {
            for j in 0..=n {
                res.set(i, j, 1.0 / (i as isize + j as isize + 1) as f64);
            }
        }
        res
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.cols * row + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[self.cols * row + col] = val;
    }

    pub fn apply(&mut self, op: &RowOperation) {
        match *op {
            RowOperation::Cmb { src, scale, dest } => self.cmb_rows(src, scale, dest),
            RowOperation::Swap(r1, r2) => self.swap_rows(r1, r2),
            RowOperation::Scale { row, scale } => self.scale_row(row, scale),
        }
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        for i in 0..self.cols() {
            let tmp = self.get(r1, i);
            self.set(r1, i, self.get(r2, i));
            self.set(r2, i, tmp);
        }
    }

    fn cmb_rows(&mut self, src: usize, scale: f64, dest: usize) {
        for i in 0..self.cols() {
            self.set(dest, i, self.get(dest, i) - scale * self.get(src, i));
        }
    }

    fn scale_row(&mut self, row: usize, scale: f64) {
        for i in 0..self.cols {
            self.set(row, i, self.get(row, i) * scale);
        }
    }

    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    pub fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.data.chunks(self.cols)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [f64]> {
        self.data.chunks_mut(self.cols)
    }

    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = &f64> {
        let start_index = row * self.cols;
        let final_index = start_index + self.cols;
        self.data[start_index..final_index].iter()
    }

    pub fn iter_col(&self, col: usize) -> impl Iterator<Item = &f64> {
        self.data.iter().enumerate().filter_map({
            let cols = self.cols;
            move |(i, v)| if i % cols != col { None } else { Some(v) }
        })
    }

    pub fn rows(&self) -> usize {
        self.data.len() / self.cols
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn k<N: Norm>(&self, inv: &Mat) -> f64 {
        N::norm(self) * N::norm(&inv)
    }

    pub fn k_approx(&self) -> Result<f64, ()> {
        assert!(self.is_square());
        let b: Vec<f64> = (1..=self.cols()).map(|a| a as f64).collect();
        let x_approx = LuDecompSolver::<Doolittle<Gaussian>>::solve(self.clone(), b.clone())?;
        let a_x_approx = self * &x_approx;
        let y_approx: Vec<f64> = b
            .iter()
            .zip(a_x_approx.iter())
            .map(|(v1, v2)| *v1 - *v2)
            .collect();
        let l2_x_approx = f64::sqrt(x_approx.iter().map(|v| v.powi(2)).sum());
        let l2_y_approx = f64::sqrt(y_approx.iter().map(|v| v.powi(2)).sum());
        Ok((l2_y_approx / l2_x_approx) * 10e16)
    }

    // maximum magnitude term in matrix
    pub fn max(&self) -> f64 {
        self.data
            .iter()
            .map(|v| f64::abs(*v))
            .fold(-1. / 0., f64::max)
    }

    fn display(&self, n_chars: usize, f: &mut fmt::Formatter) -> fmt::Result {
        fn f64_fmt_len(i: f64, len: usize) -> String {
            let mut res = format!("{:.64}", i);
            while res.len() < len {
                res.push(' ');
            }
            res[0..len].to_string()
        }

        let line_len = (((n_chars + 3) * self.cols()) as isize + -1) as usize;
        writeln!(f, "+{}+", "-".repeat(line_len))?;
        for r in self.iter() {
            write!(f, "| ")?;
            for v in r.iter() {
                write!(f, "{} | ", f64_fmt_len(*v, n_chars))?;
            }
            writeln!(f)?;
        }
        writeln!(f, "+{}+", "-".repeat(line_len))?;
        Ok(())
    }
}

impl Mul<&mut Mat> for f64 {
    type Output = ();

    fn mul(self, rhs: &mut Mat) -> <Self as Mul<&mut Mat>>::Output {
        for coef in rhs.data.iter_mut() {
            *coef *= self;
        }
    }
}

impl Mul<&Vec<f64>> for &Mat {
    type Output = Vec<f64>;

    fn mul(self, rhs: &Vec<f64>) -> <Self as Mul<&Vec<f64>>>::Output {
        assert_eq!(self.cols(), rhs.len());
        let mut res = vec![0.0; self.rows()];
        for r in 0..self.rows() {
            res[r] = rhs
                .iter()
                .zip(self.iter_row(r))
                .map(|(v1, v2)| *v1 * *v2)
                .sum();
        }
        res
    }
}

impl Mul<&Mat> for &Mat {
    type Output = Mat;

    fn mul(self, rhs: &Mat) -> <Self as Mul<&Mat>>::Output {
        assert_eq!(self.cols(), rhs.rows());
        let mut res = Mat::new(self.rows(), rhs.cols());
        for r in 0..res.rows() {
            for c in 0..res.cols() {
                res.set(
                    r,
                    c,
                    self.iter_row(r)
                        .zip(rhs.iter_col(c))
                        .map(|(v1, v2)| *v1 * *v2)
                        .sum(),
                );
            }
        }
        res
    }
}

impl Sub<&Mat> for &Mat {
    type Output = Mat;

    fn sub(self, rhs: &Mat) -> Self::Output {
        let mut res = self.clone();
        for r in 0..res.rows() {
            for c in 0..res.cols() {
                res.set(r, c, res.get(r, c) - rhs.get(r, c));
            }
        }
        res
    }
}

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const N_CHARS: usize = 10;
        self.display(N_CHARS, f)
    }
}
