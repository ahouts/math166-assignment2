use crate::mat::{Mat, RowOperation};

// trait representing the process of performing row operations
// on a matrix until it is in upper triangular form
pub trait UpperTriangle {
    fn run<F: FnMut(&RowOperation)>(m: &mut Mat, h: &mut F) -> Result<(), ()>;
}

// gaussian procedure to get into upper triangular
pub struct Gaussian;

impl UpperTriangle for Gaussian {
    fn run<F: FnMut(&RowOperation)>(m: &mut Mat, h: &mut F) -> Result<(), ()> {
        assert!(m.rows() >= m.cols());
        let n = m.rows() - 1;
        for i in 0..n {
            let p = match (i..=n).filter(|p| m.get(*p, i) != 0.0).next() {
                Some(p) => p,
                None => return Err(()),
            };
            if i != p {
                let op = RowOperation::Swap(i, p);
                m.apply(&op);
                h(&op);
            }
            for j in (i + 1)..=n {
                let op = RowOperation::Cmb {
                    src: i,
                    scale: m.get(j, i) / m.get(i, i),
                    dest: j,
                };
                m.apply(&op);
                h(&op);
            }
        }
        if m.get(n, n) == 0.0 {
            return Err(());
        }
        Ok(())
    }
}
