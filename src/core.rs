use std::{rc::{Rc, Weak}, cell::{RefCell, Ref, RefMut}, fmt};
use std::ops;
use ndarray::{ArrayD, Array, Dimension};
use num_traits::{Float};

pub struct VariableInternal<'c, A: Float> {
    pub data: ArrayD<A>,
    pub grad: Option<ArrayD<A>>,
    generation: usize,
    creator: Option<Rc<Creator<'c,A>>>,
}

impl<'c, A:Float> VariableInternal<'c,A>{
    pub fn new<D:Dimension>(data: Array<A, D>) -> Self{
        let data = data.into_dyn();
        VariableInternal {
            data,
            grad: None,
            generation: 0,
            creator: None
        }
    }

    pub fn backward(&self){
        if let Some(c) = &self.creator{
            let mut creators = vec![Rc::clone(c)];
            let mut seen_set = vec![Rc::clone(c)];
            loop{
                if creators.is_empty(){
                    break;
                }
                if let Some(c) = creators.pop(){
                    let gys = c.outputs.iter().map(|output|{
                        let output = output.upgrade().unwrap();
                        let output = output.borrow();
                        match output.grad.as_ref(){
                            Some(g) => g.clone(),
                            None => ArrayD::ones(output.data.dim()),
                        }
                    }).collect::<Vec<_>>();

                    let gxs = c.function.backward(
                        &c.inputs
                            .iter()
                            .map(|input| 
                                Ref::map(input.borrow(),|i| &i.data)
                            ).collect::<Vec<_>>(),
                        &gys
                    );
                    c.inputs.iter().zip(gxs).for_each(|(input,gx)|{
                        let mut input = input.borrow_mut();
                        input.grad = match &input.grad{
                            Some(g) => Some(g + gx.clone()),
                            None => Some(gx.clone()),
                        };
                        if let Some(ic) = &input.creator{
                            if let None = seen_set.iter().find(|s| Rc::ptr_eq(*s, ic)){
                                creators.push(Rc::clone(ic));
                                seen_set.push(Rc::clone(ic));
                                creators.sort_by(|a, b| a.generation.cmp(&b.generation));
                            }
                        }
                    });
                }
            }
        }
    }
}

#[derive(Clone)]
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

    pub fn data_mut(&self) -> RefMut<ArrayD<A>>{
        RefMut::map(self.internal.borrow_mut(), |i| &mut i.data)
    }

    pub fn grad(&self) -> Ref<Option<ArrayD<A>>>{
        Ref::map(self.internal.borrow(), |i| &i.grad)
    }

    pub fn grad_mut(&self) -> RefMut<Option<ArrayD<A>>>{
        RefMut::map(self.internal.borrow_mut(), |i| &mut i.grad)
    }

    pub fn generation(&self) -> usize{
        self.internal.borrow().generation
    }

    pub fn backward(&self){
        self.internal.borrow().backward();
    }

    pub fn cleargrad(&self){
        self.internal.borrow_mut().grad = None;
    }

    pub fn len(&self) -> usize{
        self.internal.borrow().data.len()
    }

    pub fn powf(&self, c:f64) -> Variable<'c, A>{
        powf(&self,c)
    }

    fn set_creator(&self, creator: Creator<'c, A>){
        let mut internal = self.internal.borrow_mut();
        internal.generation = &creator.generation + 1;
        internal.creator = Some(Rc::new(creator));
         
    }
}

impl<'c, A:Float + fmt::Display> fmt::Display for Variable<'c, A>{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "variable({})", self.data())
    }
}

struct Creator<'c, A: Float>{
    inputs: Vec<Rc<RefCell<VariableInternal<'c, A>>>>,
    outputs: Vec<Weak<RefCell<VariableInternal<'c, A>>>>,
    generation: usize,
    function: Rc<dyn 'c + Function<A>>,
}

pub trait Function<A: Float>{
    fn call<'c>(self, inputs: &[&Variable<'c, A>]) -> Vec<Variable<'c, A>>
    where Self: 'c + Sized
    {
        let xs = &inputs.iter().map(|input| input.data()).collect::<Vec<_>>();
        let generation = &inputs.iter().map(|input| input.generation()).max().unwrap();
        let ys = self.forward(xs);
        let outputs = ys.iter().map(|y|Variable::new(y.clone())).collect::<Vec<_>>();
        let function:Rc<dyn Function<A>>= Rc::new(self);
        outputs.iter().for_each(|output|{
            output.set_creator(Creator{
                inputs: inputs.iter().map(|input| Rc::clone(&input.internal)).collect::<Vec<_>>(),
                outputs: outputs.iter().map(|output| Rc::downgrade(&output.internal)).collect::<Vec<_>>(),
                generation: *generation,
                function: Rc::clone(&function),
            });
        });
        outputs
    }

    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>>;
    fn backward(&self, xs: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>>;
}

//arithmetic operations
//add
struct Add;
impl<A: Float> Function<A> for Add{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let y = &(*xs[0]) + &(*xs[1]);
        vec![y]
    }

    fn backward(&self, _: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

fn add<'c, A:Float>(x0: &Variable<'c, A>, x1: &Variable<'c, A>) -> Variable<'c, A>{
    Add.call(&[x0, x1])[0].clone()
}

impl <'c, A:Float> ops::Add<&Variable<'c, A>> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn add(self, x: &Variable<'c, A>) -> Variable<'c, A>{
        add(&self, &x)
    }
}

impl <'c, A:Float, D:Dimension> ops::Add<&Array<A, D>> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn add(self, x: &Array<A, D>) -> Variable<'c, A>{
        let x = Variable::new(x.clone());
        add(&self, &x)
    }
}

impl <'c, A:Float, D:Dimension> ops::Add<&Variable<'c, A>> for &Array<A, D>{
    type Output = Variable<'c, A>;
    fn add(self, x: &Variable<'c, A>) -> Variable<'c, A>{
        let x0 = Variable::new(self.clone());
        add(&x0, &x)
    }
}

impl<'c, A:Float> ops::Add<A> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn add(self, x: A) -> Self::Output {
        let x = Variable::new(Array::from_elem(self.data().dim(),x));
        add(&self, &x)
    }
}

impl<'c, A:Float> ops::Add<&Variable<'c, A>> for f64{
    type Output = Variable<'c, A>;
    fn add(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        add(&x0, &x)
    }
}

impl<'c, A:Float> ops::Add<&Variable<'c, A>> for f32{
    type Output = Variable<'c, A>;
    fn add(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        add(&x0, &x)
    }
}

//mul
struct Mul;
impl<A: Float> Function<A> for Mul{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let x0 = &(*xs[0]);
        let x1 = &(*xs[1]);
        let y = x0 * x1;
        vec![y]
    }

    fn backward(&self, xs: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        let x0 = &(*xs[0]);
        let x1 = &(*xs[1]); 
        let gy = &gys[0];
        vec![gy * x1, gy * x0]
    }
}

fn mul<'c, A:Float>(x0: &Variable<'c, A>, x1: &Variable<'c, A>) -> Variable<'c, A>{
    Mul.call(&[x0, x1])[0].clone()
}

impl<'c, A:Float> ops::Mul for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn mul(self, x: Self) -> Self::Output {
        mul(&self, &x)
    }
}

impl<'c, A:Float, D:Dimension> ops::Mul<&Array<A, D>> for &Variable<'c, A>{
    type Output = Variable<'c, A>;

    fn mul(self, x: &Array<A, D>) -> Self::Output {
        let x = Variable::new(x.clone());
        mul(&self, &x)
    }
}

impl<'c, A:Float, D:Dimension> ops::Mul<&Variable<'c, A>> for &Array<A, D>{
    type Output = Variable<'c, A>;
    fn mul(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(self.clone());
        mul(&x0, &x)
    }
}

impl<'c, A:Float> ops::Mul<A> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn mul(self, x: A) -> Self::Output {
        let x = Variable::new(Array::from_elem(self.data().dim(),x));
        mul(&self, &x)
    }
}

impl<'c, A:Float> ops::Mul<&Variable<'c, A>> for f64{
    type Output = Variable<'c, A>;
    fn mul(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        mul(&x0, &x)
    }
}

impl<'c, A:Float> ops::Mul<&Variable<'c, A>> for f32{
    type Output = Variable<'c, A>;
    fn mul(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        mul(&x0, &x)
    }
}

//neg
struct Neg;
impl<A: Float> Function<A> for Neg{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let x = &xs[0];
        let y = x.mapv(|x| -x);
        vec![y]
    }

    fn backward(&self, _: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        let gy = &gys[0];
        let gx = gy.mapv(|gy| -gy);
        vec![gx]
    }
}

fn neg<'c, A:Float>(x: &Variable<'c, A>) -> Variable<'c, A>{
    Neg.call(&[x])[0].clone()
}

impl<'c, A:Float> ops::Neg for Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn neg(self) -> Self::Output {
        neg(&self)
    }
}

//sub
struct Sub;
impl<A:Float> Function<A> for Sub{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let y = &(*xs[0]) - &(*xs[1]);
        vec![y]
    }

    fn backward(&self, _: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        vec![gys[0].clone(), -gys[0].clone()]
    }
}

fn sub<'c, A:Float>(x0: &Variable<'c, A>, x1: &Variable<'c, A>) -> Variable<'c, A>{
    Sub.call(&[x0, x1])[0].clone()
}

impl<'c, A:Float> ops::Sub for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn sub(self, x: Self) -> Self::Output {
        sub(&self, &x)
    }
}

impl<'c, A:Float, D: Dimension> ops::Sub<&Array<A, D>> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn sub(self, x: &Array<A, D>) -> Self::Output {
        let x = Variable::new(x.clone());
        sub(&self, &x)
    }
}

impl<'c, A:Float, D: Dimension> ops::Sub<&Variable<'c, A>> for &Array<A, D>{
    type Output = Variable<'c, A>;
    fn sub(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(self.clone());
        sub(&x0, &x)
    }
}

impl<'c, A:Float> ops::Sub<A> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn sub(self, x: A) -> Self::Output {
        let x = Variable::new(Array::from_elem(self.data().dim(),x));
        sub(&self, &x)
    }
}

impl<'c, A:Float> ops::Sub<&Variable<'c, A>> for f64{
    type Output = Variable<'c, A>;
    fn sub(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        sub(&x0, &x)
    }
}

impl<'c, A:Float> ops::Sub<&Variable<'c, A>> for f32{
    type Output = Variable<'c, A>;
    fn sub(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        sub(&x0, &x)
    }
}


//div
struct Div;
impl<A: Float> Function<A> for Div{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let x0 = &(*xs[0]);
        let x1 = &(*xs[1]);
        let y = x0 / x1;
        vec![y]
    }

    fn backward(&self, xs: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        let x0 = &(*xs[0]);
        let x1 = &(*xs[1]); 
        let gy = &gys[0];
        let gx0 = gy / x1;
        let gx1 = x0.mapv(|x0| -x0) / x1.mapv(|x1| x1.powi(2)) * gy; 
        vec![gx0, gx1]
    }
}

fn div<'c, A:Float>(x0: &Variable<'c, A>, x1: &Variable<'c, A>) -> Variable<'c, A>{
    Div.call(&[x0,x1])[0].clone()
}

impl<'c, A:Float> ops::Div for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn div(self, x: Self) -> Self::Output {
        div(&self, &x)
    }
}

impl<'c, A:Float, D: Dimension> ops::Div<&Array<A, D>> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn div(self, x: &Array<A, D>) -> Self::Output {
        let x = Variable::new(x.clone());
        div(&self, &x)
    }
}

impl<'c, A:Float, D: Dimension> ops::Div<&Variable<'c, A>> for &Array<A, D>{
    type Output = Variable<'c, A>;
    fn div(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(self.clone());
        div(&x0, &x)
    }
}

impl<'c, A:Float> ops::Div<A> for &Variable<'c, A>{
    type Output = Variable<'c, A>;
    fn div(self, x: A) -> Self::Output {
        let x = Variable::new(Array::from_elem(self.data().dim(),x));
        div(&self, &x)
    }
}

impl<'c, A:Float> ops::Div<&Variable<'c, A>> for f64{
    type Output = Variable<'c, A>;
    fn div(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        div(&x0, &x)
    }
}

impl<'c, A:Float> ops::Div<&Variable<'c, A>> for f32{
    type Output = Variable<'c, A>;
    fn div(self, x: &Variable<'c, A>) -> Self::Output {
        let x0 = Variable::new(Array::from_elem(x.data().dim(),A::from(self).unwrap()));
        div(&x0, &x)
    }
}

//powf
struct Powf{
    c: f64
}

impl<A: Float> Function<A> for Powf{
    fn forward(&self, xs: &[Ref<ArrayD<A>>]) -> Vec<ArrayD<A>> {
        let x = &xs[0];
        let c = A::from(self.c).unwrap(); 
        let y = x.mapv(|x| x.powf(c));
        vec![y]
    }

    fn backward(&self, xs: &[Ref<ArrayD<A>>], gys: &[ArrayD<A>]) -> Vec<ArrayD<A>> {
        let x = &xs[0];
        let gy = &gys[0];
        let c = A::from(self.c).unwrap();
        let gx = x.mapv(|x| x.powf(c - A::from(1).unwrap()) * c) * gy;
        vec![gx]
    }
}

fn powf<'c, A: Float>(input: &Variable<'c, A>, c:f64) -> Variable<'c, A>{
    Powf{c}.call(&[input])[0].clone()
}

#[test]
fn test(){
    
}