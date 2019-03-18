use crate::norm::Norm;
use std::fmt;
use std::ops::{Mul, Sub};

#[derive(Debug, Clone)]
// representation of a row operation
pub enum RowOperation {
    Swap(usize, usize),
    Cmb { src: usize, scale: f64, dest: usize },
    Scale { row: usize, scale: f64 },
}

#[derive(Debug, Clone)]
// matrix, represented by an array on the heap
pub struct Mat {
    cols: usize,
    data: Box<[f64]>,
}

impl Mat {
    // construct a new matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0);
        assert!(cols > 0);
        Mat {
            cols,
            data: vec![0.0; rows * cols].into_boxed_slice(),
        }
    }

    // construct a new identity matrix
    pub fn new_i(size: usize) -> Self {
        let mut res = Mat::new(size, size);
        let n = size - 1;
        for i in 0..=n {
            res.set(i, i, 1.0);
        }
        res
    }

    // construct a new hilbert matrix
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

    // get a value from the matrix
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.cols * row + col]
    }

    // set a value in the matrix
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[self.cols * row + col] = val;
    }

    // apply a row operation to the matrix
    pub fn apply(&mut self, op: &RowOperation) {
        match *op {
            RowOperation::Cmb { src, scale, dest } => self.cmb_rows(src, scale, dest),
            RowOperation::Swap(r1, r2) => self.swap_rows(r1, r2),
            RowOperation::Scale { row, scale } => self.scale_row(row, scale),
        }
    }

    // swap rows
    fn swap_rows(&mut self, r1: usize, r2: usize) {
        for i in 0..self.cols() {
            let tmp = self.get(r1, i);
            self.set(r1, i, self.get(r2, i));
            self.set(r2, i, tmp);
        }
    }

    // combine rows, with a scale
    fn cmb_rows(&mut self, src: usize, scale: f64, dest: usize) {
        for i in 0..self.cols() {
            self.set(dest, i, self.get(dest, i) - scale * self.get(src, i));
        }
    }

    // scale a row
    fn scale_row(&mut self, row: usize, scale: f64) {
        for i in 0..self.cols {
            self.set(row, i, self.get(row, i) * scale);
        }
    }

    // is the matrix square?
    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    // iterate through the rows
    pub fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.data.chunks(self.cols)
    }

    // iterate through the columns of the matrix, mutably
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [f64]> {
        self.data.chunks_mut(self.cols)
    }

    // iterate through values of a row of the matrix
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = &f64> {
        let start_index = row * self.cols;
        let final_index = start_index + self.cols;
        self.data[start_index..final_index].iter()
    }

    // iterate through values of a column of the matrix
    // WARNING: complexity O(n), where n is the number of cells in the matrix
    pub fn iter_col(&self, col: usize) -> impl Iterator<Item = &f64> {
        self.data.iter().enumerate().filter_map({
            let cols = self.cols;
            move |(i, v)| if i % cols != col { None } else { Some(v) }
        })
    }

    // number of rows in the matrix
    pub fn rows(&self) -> usize {
        self.data.len() / self.cols
    }

    // number of columns in the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    // calculate k for the matrix, given its inverse
    pub fn k<N: Norm>(&self, inv: &Mat) -> f64 {
        N::norm(self) * N::norm(&inv)
    }

    // maximum magnitude term in matrix
    pub fn max(&self) -> f64 {
        self.data
            .iter()
            .map(|v| f64::abs(*v))
            .fold(-1. / 0., f64::max)
    }

    // pretty print the matrix
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

// float * matrix
impl Mul<&Mat> for f64 {
    type Output = Mat;

    fn mul(self, rhs: &Mat) -> <Self as Mul<&Mat>>::Output {
        let mut res = rhs.clone();
        for coef in res.data.iter_mut() {
            *coef *= self;
        }
        res
    }
}

// matrix * vector
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

// matrix * matrix
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

// matrix - matrix
impl Sub<&Mat> for &Mat {
    type Output = Mat;

    fn sub(self, rhs: &Mat) -> Self::Output {
        assert_eq!(self.rows(), rhs.rows());
        assert_eq!(self.cols(), rhs.cols());

        let mut res = self.clone();
        for r in 0..res.rows() {
            for c in 0..res.cols() {
                res.set(r, c, res.get(r, c) - rhs.get(r, c));
            }
        }
        res
    }
}

// allow matrices to be printed by println
impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const N_CHARS: usize = 10;
        self.display(N_CHARS, f)
    }
}
