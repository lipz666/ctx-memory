//! Embed the git commit the binary was built from (`ctx --version`, status, health), so
//! an installed daemon can be matched to its source.
use std::process::Command;

fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let hash = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=CTX_GIT_HASH={hash}");
    // Rebuild when HEAD moves: the HEAD file itself and the branch it points to.
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
        println!("cargo:rerun-if-changed=.git/{branch}");
    }
}
