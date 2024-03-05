extern crate anyhow;
extern crate tch;

use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            IMAGE_DIM,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

pub fn run() -> Result<()> {
    let mnist = tch::vision::mnist::load_dir("data")?;
    let device = Device::cuda_if_available();
    println!("{:?}", device);

    let vs  = nn::VarStore::new(device);
    let net = net(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..200 {
        let loss = net
            .forward(&mnist.train_images.to(device))
            .cross_entropy_for_logits(&mnist.train_labels.to(device));
        opt.backward_step(&loss);
        

        if epoch % 10 == 0 {
            let test_accuracy = net
            .forward(&mnist.test_images.to(device))
            .accuracy_for_logits(&mnist.test_labels.to(device));

            println!("epoch: {:4}, train loss: {:8.5}, test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                1. * f64::from(&test_accuracy),
            )
        }
    }


    Ok(())
}

fn main() {
    let _ = run();
}