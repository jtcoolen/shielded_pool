use std::num::NonZeroUsize;
use std::sync::OnceLock;

use rayon::ThreadPoolBuilder;

const ENV_RAYON_THREADS: &str = "SHIELDED_POOL_RAYON_THREADS";
const ENV_NUMA_NODE: &str = "SHIELDED_POOL_NUMA_NODE";

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
}

pub fn configure_global_rayon_pool() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let cfg = NumaConfig::from_env();
        let mut candidate_cores = candidate_cores(cfg.target_node);
        let default_threads = if candidate_cores.is_empty() {
            available_parallelism()
        } else {
            candidate_cores.len()
        };
        let num_threads = cfg.requested_threads.unwrap_or(default_threads).max(1);

        if !candidate_cores.is_empty() {
            candidate_cores = expand_core_plan(candidate_cores, num_threads);
        }

        let mut builder = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("ivc-rayon-{i}"));

        if !candidate_cores.is_empty() {
            let pin_plan = candidate_cores;
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

fn candidate_cores(target_node: Option<usize>) -> Vec<core_affinity::CoreId> {
    let all_cores = core_affinity::get_core_ids().unwrap_or_default();
    if all_cores.is_empty() {
        return all_cores;
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(groups) = linux_numa_core_groups(&all_cores) {
            if let Some(node) = target_node {
                if let Some((_, cores)) = groups.iter().find(|(id, _)| *id == node) {
                    return cores.clone();
                }
            }
            return groups.into_iter().flat_map(|(_, cores)| cores).collect();
        }
    }

    let _ = target_node;
    all_cores
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
