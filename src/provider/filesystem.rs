use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::UNIX_EPOCH;
use std::{env, fs};

use crate::config::{FilesystemProviderConfig, GlobalAgentConfig};
use crate::data::extended_decision::{FileContent, FileDecisionGraph, FileTestContent};
use crate::data::release_data::ReleaseData;
use crate::immutable_loader::{ImmutableLoader, collect_examples};
use crate::provider::{
    AgentData, AgentDataProvider, FailedProjectsRegistry, Project, ProjectDiff,
};
use anyhow::Context;
use itertools::Itertools;
use tokio::task;
use walkdir::WalkDir;

#[derive(Debug)]
pub struct FilesystemProvider {
    root_dir: PathBuf,
}

impl FilesystemProvider {
    pub fn new(config: &FilesystemProviderConfig, _: Arc<GlobalAgentConfig>) -> Self {
        let root = env::current_dir()
            .expect("Current directory is available")
            .join(config.root_dir.as_str())
            .to_path_buf();

        Self { root_dir: root }
    }
}

impl AgentDataProvider for FilesystemProvider {
    fn load_data(
        &self,
        data: Arc<AgentData>,
    ) -> impl Future<Output = anyhow::Result<Vec<ProjectDiff>>> + Send + 'static {
        let root = self.root_dir.clone();

        async move {
            let blocking_data = data.clone();
            let (loaded, removed) = task::spawn_blocking(move || {
                let data = blocking_data;
                let directory = match fs::read_dir(root.clone()) {
                    Ok(dir) => dir,
                    Err(error) => {
                        println!("[FS - Skip] Failed to read directory: {}", error);
                        return (Vec::new(), Vec::new());
                    }
                };

                let paths = directory
                    .into_iter()
                    .filter_map(|d| {
                        let Ok(entry) = d else {
                            return None;
                        };

                        let Ok(meta) = entry.metadata() else {
                            return None;
                        };

                        meta.is_dir().then_some(entry.path().to_path_buf())
                    })
                    .collect::<Vec<PathBuf>>();

                let mut seen: HashSet<String> = HashSet::new();
                let mut loaded: Vec<(String, Arc<Project>, bool)> = Vec::new();

                for directory in paths {
                    let relative_path = match directory.strip_prefix(root.clone()) {
                        Ok(ok) => ok.to_string_lossy().to_string(),
                        Err(err) => {
                            tracing::error!(
                                "[FS - Skip] failed to strip prefix on {}: {}",
                                directory.display(),
                                err
                            );
                            continue;
                        }
                    };
                    seen.insert(relative_path.clone());

                    let hash = match fingerprint(&directory) {
                        Ok(hash) => hash,
                        Err(err) => {
                            tracing::error!(
                                "[FS - Skip] failed to fingerprint {}: {}",
                                directory.display(),
                                err
                            );
                            continue;
                        }
                    };

                    let existing = data.projects.get(&relative_path);
                    if existing
                        .as_ref()
                        .is_some_and(|p| p.content_hash.as_deref() == Some(hash.as_slice()))
                        || FailedProjectsRegistry::has_failed(Some(hash.as_slice()))
                    {
                        // Unchanged since the last poll (or known-broken at this exact content).
                        continue;
                    }
                    let is_new = existing.is_none();
                    drop(existing);

                    match load_from_directory(&directory, Some(hash.clone())) {
                        Ok(project) => loaded.push((relative_path, Arc::new(project), is_new)),
                        Err(err) => {
                            tracing::error!(
                                "[FS - Skip] failed to load project from directory {}: {}",
                                directory.display(),
                                err
                            );
                            FailedProjectsRegistry::insert(hash);
                        }
                    }
                }

                let removed = data
                    .projects
                    .iter()
                    .map(|e| e.key().to_string())
                    .filter(|key| !seen.contains(key))
                    .collect::<Vec<_>>();

                (loaded, removed)
            })
            .await?;

            let mut diff = Vec::with_capacity(loaded.len() + removed.len());
            for key in removed {
                data.projects.remove(&key);
                diff.push(ProjectDiff::Removed(key));
            }
            for (key, project, is_new) in loaded {
                data.projects.insert(key.clone(), project);
                diff.push(if is_new {
                    ProjectDiff::Created(key)
                } else {
                    ProjectDiff::Updated(key)
                });
            }

            Ok(diff)
        }
    }
}

/// Cheap change detector for a project directory: a hash over every file's relative
/// path, size and modified time. Equal fingerprints mean there is nothing to reload.
fn fingerprint(root: &Path) -> anyhow::Result<Vec<u8>> {
    let mut hasher = DefaultHasher::new();
    for entry in WalkDir::new(root).sort_by_file_name() {
        let entry = entry.context("failed to walk directory")?;
        if !entry.file_type().is_file() {
            continue;
        }
        let meta = entry.metadata().context("failed to read file metadata")?;
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        entry.path().strip_prefix(root).unwrap_or(entry.path()).hash(&mut hasher);
        meta.len().hash(&mut hasher);
        modified.hash(&mut hasher);
    }
    Ok(hasher.finish().to_le_bytes().to_vec())
}

fn load_from_directory(
    root: &PathBuf,
    content_hash: Option<Vec<u8>>,
) -> anyhow::Result<Project> {
    let files = WalkDir::new(root.clone())
        .into_iter()
        .filter_ok(|d| d.file_type().is_file())
        .collect::<Result<Vec<_>, _>>()
        .context("failed to load files")?;

    let project_json_path = Some(root.join(".config").join("project.json"));
    let release_data = project_json_path
        .map(|entry| {
            let file_reader = File::open(entry).ok()?;
            ReleaseData::from_json_reader(file_reader)
        })
        .flatten();

    let mut graphs: HashMap<String, FileDecisionGraph> = HashMap::new();
    let mut test_files: Vec<(String, FileTestContent)> = Vec::new();

    for entry in files.iter() {
        let Ok(relative_path) = entry.path().strip_prefix(&root) else {
            continue;
        };
        if relative_path.starts_with(".config") {
            continue;
        }

        let path = relative_path.to_string_lossy().to_string();
        let file_reader = File::open(entry.path()).context("failed to open file")?;
        let content: FileContent = serde_json::from_reader(file_reader)
            .with_context(|| format!("failed to parse decision content for file {path}"))?;

        match content {
            FileContent::Graph(mut graph) => {
                // Keep the original-cased path for display, but key the map by the
                // lowercased path — `ImmutableLoader::load` lowercases lookup keys,
                // so a mixed-case key here would be unreachable at evaluation time.
                graph.meta.display_path = Some(Arc::from(path.as_str()));
                graphs.insert(path.to_lowercase(), graph);
            }
            FileContent::Test(test) => test_files.push((path, test)),
            FileContent::Unknown => {}
        }
    }

    let examples = collect_examples(test_files);

    Ok(Project {
        engine: ImmutableLoader::new(graphs, examples, release_data).into_engine(),
        content_hash,
        rules_spec: OnceLock::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_ext::EngineExtension;

    /// Mixed-case file names must be reachable at evaluation time (which
    /// lowercases lookup keys via `ImmutableLoader::load`), while the
    /// original casing is preserved for display in the rules OpenAPI spec.
    #[test]
    fn load_from_directory_lowercases_keys_but_keeps_display_path() {
        let dir = env::temp_dir().join(format!(
            "gorules-agent-fs-test-{}-{}",
            std::process::id(),
            "load_from_directory_lowercases_keys_but_keeps_display_path"
        ));
        fs::create_dir_all(&dir).expect("failed to create temp test dir");

        let graph_json = r#"{
            "contentType": "graph",
            "nodes": [
                { "id": "in", "name": "request", "type": "inputNode" },
                { "id": "out", "name": "response", "type": "outputNode" }
            ],
            "edges": [{ "id": "e1", "sourceId": "in", "targetId": "out" }]
        }"#;
        fs::write(dir.join("Mixed Case Rule"), graph_json).expect("failed to write graph fixture");

        let result = load_from_directory(&dir, None);

        // Clean up before asserting so the temp dir is never left behind on failure.
        let _ = fs::remove_dir_all(&dir);

        let project = result.expect("failed to load project from directory");

        let keys = project.engine.decision_keys();
        assert!(
            keys.contains(&"mixed case rule".to_string()),
            "expected lowercased key in decision_keys(), got {keys:?}"
        );

        let entries = project.engine.spec_entries();
        let entry = entries
            .iter()
            .find(|e| e.path.as_ref() == "Mixed Case Rule")
            .expect("expected an entry with the original-cased display path");
        assert_eq!(entry.path.as_ref(), "Mixed Case Rule");
    }

    #[test]
    fn fingerprint_changes_only_when_files_change() {
        let dir = env::temp_dir().join(format!("gorules-agent-fs-fp-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("failed to create temp test dir");
        fs::write(dir.join("a.json"), "{}").expect("write");

        let first = fingerprint(&dir).expect("fingerprint");
        assert_eq!(first, fingerprint(&dir).expect("fingerprint"));

        fs::write(dir.join("b.json"), "{}").expect("write");
        let after_add = fingerprint(&dir).expect("fingerprint");
        assert_ne!(first, after_add);

        fs::remove_file(dir.join("b.json")).expect("remove");
        let after_remove = fingerprint(&dir).expect("fingerprint");
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(first, after_remove);
    }
}
