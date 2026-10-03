//! Full tool outputs, stored by content hash under `CCX_HOME/objects` so a shortened
//! output can be read back with `ccx_expand`. Files are readable by the owner only.
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

pub struct Objects {
    dir: PathBuf,
}

pub fn id_of(text: &str) -> String {
    format!("x{}", &hex::encode(Sha256::digest(text.as_bytes()))[..12])
}

impl Objects {
    pub fn open(home: &Path) -> std::io::Result<Self> {
        let dir = home.join("objects");
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Store `text` (once) and return its id.
    pub fn put(&self, text: &str) -> std::io::Result<String> {
        let id = id_of(text);
        let path = self.dir.join(&id);
        if !path.exists() {
            let tmp = self.dir.join(format!("{id}.tmp"));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create(true).truncate(true);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
            options.open(&tmp)?.write_all(text.as_bytes())?;
            std::fs::rename(tmp, path)?;
        }
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Option<String> {
        if !id.starts_with('x') || id.len() != 13 || !id[1..].chars().all(|c| c.is_ascii_hexdigit())
        {
            return None;
        }
        std::fs::read_to_string(self.dir.join(id)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_rejects_paths() {
        let dir = tempfile::tempdir().unwrap();
        let objects = Objects::open(dir.path()).unwrap();
        let id = objects.put("hello").unwrap();
        assert_eq!(id, objects.put("hello").unwrap());
        assert_eq!(objects.get(&id).as_deref(), Some("hello"));
        assert_eq!(objects.get("../steps.jsonl"), None);
        assert_eq!(objects.get("x000000000000"), None);
    }
}
