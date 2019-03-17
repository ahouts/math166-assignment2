use crate::mat::Mat;

pub trait Norm {
    fn norm(m: &Mat) -> f64;
}

pub struct LInf;

impl Norm for LInf {
    fn norm(m: &Mat) -> f64 {
        let mut res = 0.0;
        for c in 0..m.cols() {
            let mut curr = 0.0;
            for r in 0..m.rows() {
                curr += f64::abs(m.get(r, c));
            }
            if curr > res {
                res = curr;
            }
        }
        res
    }
}
