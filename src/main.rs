use deepzero_rust::core::{Variable};

fn main() {
    let a = Variable::new(ndarray::arr1(&[3.0, 2.0, 5.0]));
    let b = Variable::new(ndarray::arr1(&[2.0]));
    let c = 2.0f64 / &a;
    c.backward();
    println!("{:?}", c.data());
}
