use ndarray::ArrayD;
use deepzero_rust::core::{Variable, square, exp};

fn main() {
    let data:ArrayD<f64> = ndarray::arr1(&[0.5]).into_dyn();
    let x = Variable::new(data);
    let a = square(&x);
    let b = exp(&a);
    let y = square(&b);
    y.backward();
    print!("{:?}",x.grad());
}
