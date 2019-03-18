use crate::mat::{Mat, RowOperation};

// trait representing the process of performing row operations
// on an upper triangular matrix until it is the identity matrix
pub trait ReduceUpper {
    fn run<F: FnMut(&RowOperation)>(m: &mut Mat, h: &mut F) -> Result<(), ()>;
}

// trivial reduce method
pub struct BasicReduceUpper;

impl ReduceUpper for BasicReduceUpper {
    fn run<F: FnMut(&RowOperation)>(m: &mut Mat, h: &mut F) -> Result<(), ()> {
        assert!(m.rows() >= m.cols());
        let n = m.rows() - 1;
        for i in (0..=n).rev() {
            for j in 0..i {
                let op = RowOperation::Cmb {
                    src: i,
                    scale: m.get(j, i) / m.get(i, i),
                    dest: j,
                };
                m.apply(&op);
                h(&op);
            }
        }
        for i in 0..=n {
            let op = RowOperation::Scale {
                row: i,
                scale: 1.0 / m.get(i, i),
            };
            m.apply(&op);
            h(&op);
        }
        Ok(())
    }
}
