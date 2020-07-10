use feat_extr::config::{arg_matches, Config};
use feat_extr::run;

fn main() {
    run(Config::from_arg_matches(&arg_matches()));
}
