use std::cell::Ref;
use ndarray::ArrayD;
use num_traits::Float;
use crate::core::{Variable, Function};

pub struct Square;
impl<A: Float> Function<A> for Square{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let x = &xs[0];
        let y = x.mapv(|x| x.powi(2));
        vec![y]
    }

    fn backward(&self, xs: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        let x = &xs[0];
        let gy = &gys[0];
        let two = A::from(2).unwrap();
        let gx = x.mapv(|x| x * two) * gy.mapv(|gy| gy);
        vec![gx]
    }
}

pub fn square<'c,A: Float>(input: &Variable<'c,A>) -> Variable<'c, A>{
    Square.call(&[input])[0].clone()
}

//exp
pub struct Exp;
impl<A: Float> Function<A> for Exp{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let x = &xs[0];
        let y = x.mapv(|x| x.exp());
        vec![y]
    }

    fn backward(&self, xs: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        let x = &xs[0];
        let gy = &gys[0];
        let gx = x.mapv(|x| x.exp()) * gy.mapv(|gy| gy);
        vec![gx]
    }
}

pub fn exp<'c,A: Float>(input: &Variable<'c,A>) -> Variable<'c,A>{
    Exp.call(&[input])[0].clone()
}