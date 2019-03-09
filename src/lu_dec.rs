use crate::mat::Mat;

pub trait LuDec {
    fn dec(m: Mat) -> (Mat, Mat);
}

pub struct Dolittle;

impl LuDec for Dolittle {
    fn dec(m: Mat) -> (Mat, Mat) {
        assert!(m.is_square());
        (m.clone(), m)
    }
}
