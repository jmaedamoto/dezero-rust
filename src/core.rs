use std::{rc::Rc, cell::{RefCell, Ref}};

use ndarray::{ArrayD, Array, Dimension};
use num_traits::{Float};

pub struct VariableInternal<'c, A: Float> {
    pub data: ArrayD<A>,
    pub grad: Option<ArrayD<A>>,
    creator: Option<Rc<Creator<'c,A>>>,
}

impl<'c, A:Float> VariableInternal<'c,A>{
    pub fn new<D:Dimension>(data: Array<A, D>) -> Self{
        VariableInternal {
            data:data.into_dyn(),
            grad: None,
            creator: None
        }
    }

    pub fn backward(&mut self){
        let gy = match &self.grad{
            Some(g) => g.clone(),
            None => {
                let g = ArrayD::ones(self.data.dim());
                self.grad = Some(g.clone());
                g
            }
        };

        if let Some(c) = &self.creator{
            let creator = Rc::clone(c);
            let mut input = creator.input.borrow_mut();
            let gx = creator.function.backward(&input.data, &gy);
            input.grad = Some(gx);
            input.backward();
        }
    }
}

pub struct Variable<'c, A: Float> {
    internal: Rc<RefCell<VariableInternal<'c, A>>>
}

impl<'c, A: Float> Variable<'c, A>{
    pub fn new<D:Dimension>(data: Array<A, D>) -> Self{
        let internal = VariableInternal::new(data);
        Variable{
            internal: Rc::new(RefCell::new(internal))
        }
    }
    
    pub fn data(&self) -> Ref<ArrayD<A>>{
        Ref::map(self.internal.borrow(), |i| &i.data)
    }

    pub fn grad(&self) -> Ref<Option<ArrayD<A>>>{
        Ref::map(self.internal.borrow(), |i| &i.grad)
    }

    pub fn backward(&self){
        self.internal.borrow_mut().backward();
    }
}

struct Creator<'c, A: Float>{
    input: Rc<RefCell<VariableInternal<'c, A>>>,
    function: Box<dyn 'c + Function<A>>,
}

trait Function<A: Float>{
    fn call<'c>(self, input: &Variable<'c, A>) -> Variable<'c, A>
    where Self: 'c + Sized
    {
        let x = &input.data();
        let y = self.forward(x);
        let output = Variable::new(y);
        output.internal.borrow_mut().creator = Some(Rc::new(Creator{
            input: Rc::clone(&input.internal),
            function: Box::new(self)
        }));
        output
    }

    fn forward(&self, x: &ArrayD<A>) -> ArrayD<A>;
    fn backward(&self, x: &ArrayD<A>, gy: &ArrayD<A>) -> ArrayD<A>;
}

pub struct Square;
impl<A: Float> Function<A> for Square{
    fn forward(&self, x: &ArrayD<A>) -> ArrayD<A> {
        x.mapv(|x| x.powi(2))
    }

    fn backward(&self, x: &ArrayD<A>, gy: &ArrayD<A>) -> ArrayD<A> {
        let two = A::from(2).unwrap();
        (x * gy).mapv(|x| x * two)
    }
}

pub fn square<'c,A: Float>(input: &Variable<'c,A>) -> Variable<'c, A>{
    Square.call(input)
}

pub struct Exp;
impl<A: Float> Function<A> for Exp{
    fn forward(&self, x: &ArrayD<A>) -> ArrayD<A> {
        x.mapv(|x| x.exp())
    }

    fn backward(&self, x: &ArrayD<A>, gy: &ArrayD<A>) -> ArrayD<A> {
        x.mapv(|x| x.exp()) * gy
    }
}

pub fn exp<'c,A: Float>(input: &Variable<'c,A>) -> Variable<'c,A>{
    Exp.call(input)
}

#[test]
fn test(){
    let data:ArrayD<f64> = ndarray::arr1(&[0.5]).into_dyn();
    let x = Variable::new(data);
    let y = square(&x);
    assert_eq!(y.data()[0], 100.0);
}