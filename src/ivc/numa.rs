use std::num::NonZeroUsize;
use std::sync::OnceLock;

use rayon::ThreadPoolBuilder;

const ENV_RAYON_THREADS: &str = "SHIELDED_POOL_RAYON_THREADS";
const ENV_NUMA_NODE: &str = "SHIELDED_POOL_NUMA_NODE";
const ENV_NUMA_NODES: &str = "SHIELDED_POOL_NUMA_NODES";

#[derive(Clone, Copy, Debug)]
struct NumaConfig {
    requested_threads: Option<usize>,
    target_node: Option<usize>,
}

impl NumaConfig {
    fn from_env() -> Self {
        Self {
            requested_threads: std::env::var(ENV_RAYON_THREADS)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0),
            target_node: std::env::var(ENV_NUMA_NODE)
                .ok()
                .and_then(|v| v.parse::<usize>().ok()),
        }
    }

    fn target_nodes(self) -> Option<Vec<usize>> {
        let multi = std::env::var(ENV_NUMA_NODES)
            .ok()
            .map(|raw| {
                raw.split(',')
                    .filter_map(|x| x.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|nodes| !nodes.is_empty());

        multi.or_else(|| self.target_node.map(|n| vec![n]))
    }
}

pub fn configure_global_rayon_pool(max_threads_hint: Option<usize>) {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let cfg = NumaConfig::from_env();
        let target_nodes = cfg.target_nodes();
        let max_threads_hint = max_threads_hint.filter(|&n| n > 0);
        let mut candidate_cores = target_nodes.as_deref().and_then(candidate_cores_for_nodes);
        let default_threads = candidate_cores
            .as_ref()
            .map(|cores| cores.len())
            .unwrap_or_else(available_parallelism);
        let mut num_threads = cfg.requested_threads.unwrap_or(default_threads).max(1);
        if cfg.requested_threads.is_none() {
            if let Some(hint) = max_threads_hint {
                num_threads = num_threads.min(hint.max(1));
            }
        }
        if let Some(cores) = candidate_cores.as_ref() {
            num_threads = num_threads.min(cores.len().max(1));
        }

        if let Some(cores) = candidate_cores.as_mut() {
            *cores = expand_core_plan(cores.clone(), num_threads);
        }

        let mut builder = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("ivc-rayon-{i}"));

        if let Some(pin_plan) = candidate_cores {
            builder = builder.start_handler(move |idx| {
                let core = pin_plan[idx % pin_plan.len()];
                let _ = core_affinity::set_for_current(core);
            });
        }

        let _ = builder.build_global();
    });
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1)
}

fn candidate_cores_for_nodes(target_nodes: &[usize]) -> Option<Vec<core_affinity::CoreId>> {
    if target_nodes.is_empty() {
        return None;
    }

    let all_cores = core_affinity::get_core_ids().unwrap_or_default();
    if all_cores.is_empty() {
        return None;
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(groups) = linux_numa_core_groups(&all_cores) {
            let mut per_node = Vec::new();
            for target in target_nodes {
                if let Some((_, cores)) = groups.iter().find(|(id, _)| id == target) {
                    per_node.push(cores.clone());
                }
            }
            return interleave_cores(per_node);
        }
        return None;
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = target_nodes;
        Some(all_cores)
    }
}

#[cfg(target_os = "linux")]
fn interleave_cores(
    per_node: Vec<Vec<core_affinity::CoreId>>,
) -> Option<Vec<core_affinity::CoreId>> {
    if per_node.is_empty() {
        return None;
    }

    let max_len = per_node.iter().map(Vec::len).max().unwrap_or(0);
    if max_len == 0 {
        return None;
    }

    let mut out = Vec::new();
    for i in 0..max_len {
        for cores in &per_node {
            if let Some(core) = cores.get(i) {
                out.push(*core);
            }
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn expand_core_plan(
    mut cores: Vec<core_affinity::CoreId>,
    num_threads: usize,
) -> Vec<core_affinity::CoreId> {
    if cores.len() >= num_threads {
        cores.truncate(num_threads);
        return cores;
    }

    let base = cores.clone();
    while cores.len() < num_threads {
        cores.push(base[cores.len() % base.len()]);
    }
    cores
}

#[cfg(target_os = "linux")]
fn linux_numa_core_groups(
    all_cores: &[core_affinity::CoreId],
) -> Option<Vec<(usize, Vec<core_affinity::CoreId>)>> {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    let available: BTreeMap<usize, core_affinity::CoreId> =
        all_cores.iter().map(|c| (c.id, *c)).collect();
    let mut groups = Vec::new();
    let node_dir = Path::new("/sys/devices/system/node");
    let entries = fs::read_dir(node_dir).ok()?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(node_id) = name
            .strip_prefix("node")
            .and_then(|v| v.parse::<usize>().ok())
        else {
            continue;
        };

        let cpulist_path = entry.path().join("cpulist");
        let Ok(cpulist) = fs::read_to_string(cpulist_path) else {
            continue;
        };

        let mut cores = Vec::new();
        for cpu in parse_cpu_list(&cpulist) {
            if let Some(core) = available.get(&cpu) {
                cores.push(*core);
            }
        }

        if !cores.is_empty() {
            groups.push((node_id, cores));
        }
    }

    groups.sort_by_key(|(node_id, _)| *node_id);
    if groups.is_empty() {
        None
    } else {
        Some(groups)
    }
}

#[cfg(target_os = "linux")]
fn parse_cpu_list(value: &str) -> Vec<usize> {
    let mut cpus = Vec::new();

    for token in value.trim().split(',').filter(|t| !t.is_empty()) {
        if let Some((start, end)) = token.split_once('-') {
            let Ok(start) = start.parse::<usize>() else {
                continue;
            };
            let Ok(end) = end.parse::<usize>() else {
                continue;
            };
            if start <= end {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = token.parse::<usize>() {
            cpus.push(cpu);
        }
    }

    cpus
}
