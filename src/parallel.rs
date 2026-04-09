use std::sync::{Arc, OnceLock};

use core_affinity::CoreId;
#[cfg(target_os = "linux")]
use std::collections::{HashMap, HashSet};

static RAYON_CONFIG: OnceLock<()> = OnceLock::new();

pub fn configure_rayon_global() {
    RAYON_CONFIG.get_or_init(|| {
        let Some(mut core_ids) = core_affinity::get_core_ids() else {
            return;
        };
        if core_ids.is_empty() {
            return;
        }

        if numa_awareness_enabled() {
            if let Some(reordered) = linux_numa_interleaved_cores(&core_ids) {
                if !reordered.is_empty() {
                    core_ids = reordered;
                }
            }
        }

        let thread_count = rayon_thread_count(core_ids.len());
        let pinned_cores = Arc::new(core_ids);
        let start_handler_cores = Arc::clone(&pinned_cores);

        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .start_handler(move |idx| {
                if let Some(core_id) = start_handler_cores.get(idx % start_handler_cores.len()) {
                    let _ = core_affinity::set_for_current(*core_id);
                }
            })
            .build_global();

        if let Err(err) = result {
            eprintln!("rayon global pool configuration skipped: {err}");
        }
    });
}

fn rayon_thread_count(max_threads: usize) -> usize {
    std::env::var("SHIELDED_POOL_RAYON_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .map(|n| n.min(max_threads))
        .unwrap_or(max_threads)
}

fn numa_awareness_enabled() -> bool {
    !matches!(
        std::env::var("SHIELDED_POOL_NUMA_AWARE")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("0") | Some("false") | Some("no") | Some("off")
    )
}

#[cfg(target_os = "linux")]
fn linux_numa_interleaved_cores(available: &[CoreId]) -> Option<Vec<CoreId>> {
    let available_map: HashMap<usize, CoreId> = available.iter().map(|c| (c.id, *c)).collect();
    let available_set: HashSet<usize> = available_map.keys().copied().collect();

    let mut per_node: Vec<Vec<usize>> = std::fs::read_dir("/sys/devices/system/node")
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let file_name = entry.file_name();
            let name = file_name.to_str()?;
            if !name.starts_with("node") {
                return None;
            }
            let cpulist_path = entry.path().join("cpulist");
            let cpulist = std::fs::read_to_string(cpulist_path).ok()?;
            let mut cpus: Vec<usize> = parse_linux_cpu_list(&cpulist)?
                .into_iter()
                .filter(|cpu| available_set.contains(cpu))
                .collect();
            cpus.sort_unstable();
            if cpus.is_empty() { None } else { Some(cpus) }
        })
        .collect();

    if per_node.len() < 2 {
        return None;
    }
    per_node.sort_by_key(|cpus| cpus[0]);

    let mut interleaved = Vec::with_capacity(available.len());
    let mut offsets = vec![0usize; per_node.len()];

    loop {
        let mut pushed = false;
        for (node_idx, node_cpus) in per_node.iter().enumerate() {
            if let Some(cpu_id) = node_cpus.get(offsets[node_idx]) {
                interleaved.push(*cpu_id);
                offsets[node_idx] += 1;
                pushed = true;
            }
        }
        if !pushed {
            break;
        }
    }

    let mut dedup = HashSet::with_capacity(interleaved.len());
    let reordered: Vec<CoreId> = interleaved
        .into_iter()
        .filter(|cpu_id| dedup.insert(*cpu_id))
        .filter_map(|cpu_id| available_map.get(&cpu_id).copied())
        .collect();

    if reordered.is_empty() {
        None
    } else {
        Some(reordered)
    }
}

#[cfg(not(target_os = "linux"))]
fn linux_numa_interleaved_cores(_available: &[CoreId]) -> Option<Vec<CoreId>> {
    None
}

#[cfg(target_os = "linux")]
fn parse_linux_cpu_list(raw: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for segment in raw.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = segment.split_once('-') {
            let lo = start.trim().parse::<usize>().ok()?;
            let hi = end.trim().parse::<usize>().ok()?;
            if lo > hi {
                return None;
            }
            cpus.extend(lo..=hi);
        } else {
            cpus.push(segment.trim().parse::<usize>().ok()?);
        }
    }
    Some(cpus)
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_cpu_list() {
        let parsed = super::parse_linux_cpu_list("0-2,5,8-9").unwrap();
        assert_eq!(parsed, vec![0, 1, 2, 5, 8, 9]);
    }
}
