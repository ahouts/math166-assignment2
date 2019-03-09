use std::ops::Mul;

#[derive(Debug, Clone)]
pub enum RowOperation {
    Swap(usize, usize),
    Scale { src: usize, scale: f64, dest: usize },
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
            data: Vec::with_capacity(rows * cols).into_boxed_slice(),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.cols * row + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[self.cols * row + col] = val;
    }

    pub fn apply(&mut self, op: &RowOperation) {
        match op {
            &RowOperation::Scale { src, scale, dest } => self.scale_row(src, scale, dest),
            &RowOperation::Swap(r1, r2) => self.swap_row(r1, r2),
        }
    }

    fn swap_row(&mut self, r1: usize, r2: usize) {
        for i in 0..self.cols() {
            let tmp = self.get(r1, i);
            self.set(r1, i, self.get(r2, i));
            self.set(r2, i, tmp);
        }
    }

    fn scale_row(&mut self, src: usize, scale: f64, dest: usize) {
        for i in 0..self.cols() {
            self.set(dest, i, scale * self.get(src, i));
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
}

impl Mul<&mut Mat> for f64 {
    type Output = ();

    fn mul(self, rhs: &mut Mat) -> <Self as Mul<&mut Mat>>::Output {
        for coef in rhs.data.iter_mut() {
            *coef *= self;
        }
    }
}

impl Mul<&Mat> for &Mat {
    type Output = Mat;

    fn mul(self, rhs: &Mat) -> <Self as Mul<&Mat>>::Output {
        assert_eq!(self.cols(), rhs.rows());
        let mut res = Mat::new(self.rows(), rhs.cols());
        for (r, row) in res.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = self
                    .iter_row(r)
                    .zip(rhs.iter_col(c))
                    .map(|(v1, v2)| *v1 * *v2)
                    .sum();
            }
        }
        res
    }
}
