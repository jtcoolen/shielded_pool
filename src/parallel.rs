use std::sync::{Arc, OnceLock};

use core_affinity::CoreId;
use rayon::prelude::*;
#[cfg(target_os = "linux")]
use std::collections::{HashMap, HashSet};

static RAYON_CONFIG: OnceLock<()> = OnceLock::new();
static NUMA_POOLS: OnceLock<Option<Vec<NumaPool>>> = OnceLock::new();

struct NumaPool {
    pool: rayon::ThreadPool,
}

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

pub fn static_block_size(len: usize) -> usize {
    static_block_size_with_threads(len, rayon::current_num_threads())
}

pub fn numa_parallel_map<T, R, E, F>(items: Vec<T>, f: F) -> Result<Vec<R>, E>
where
    T: Send,
    R: Send,
    E: Send,
    F: Fn(T) -> Result<R, E> + Sync + Send,
{
    if items.is_empty() {
        return Ok(vec![]);
    }

    let Some(pools) = numa_pools() else {
        return fallback_parallel_map(items, f);
    };

    let chunk_size = items.len().div_ceil(pools.len().max(1));
    let mut chunked: Vec<(usize, Vec<T>)> = Vec::new();
    let mut iter = items.into_iter();
    let mut chunk_idx = 0usize;

    loop {
        let chunk: Vec<T> = iter.by_ref().take(chunk_size).collect();
        if chunk.is_empty() {
            break;
        }
        chunked.push((chunk_idx, chunk));
        chunk_idx += 1;
    }

    let mut joined: Vec<(usize, Result<Vec<R>, E>)> = Vec::with_capacity(chunked.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunked.len());
        for (idx, chunk) in chunked {
            let pool = &pools[idx % pools.len()];
            let f_ref = &f;
            handles.push(scope.spawn(move || {
                let out = pool.pool.install(|| {
                    let mut out = Vec::with_capacity(chunk.len());
                    for item in chunk {
                        out.push(f_ref(item)?);
                    }
                    Ok::<Vec<R>, E>(out)
                });
                (idx, out)
            }));
        }
        for handle in handles {
            joined.push(
                handle
                    .join()
                    .expect("numa worker thread panicked while processing chunk"),
            );
        }
    });

    joined.sort_by_key(|(idx, _)| *idx);
    let mut out = Vec::new();
    for (_, res) in joined {
        out.extend(res?);
    }
    Ok(out)
}

fn fallback_parallel_map<T, R, E, F>(items: Vec<T>, f: F) -> Result<Vec<R>, E>
where
    T: Send,
    R: Send,
    E: Send,
    F: Fn(T) -> Result<R, E> + Sync + Send,
{
    let block = static_block_size(items.len());
    items
        .into_par_iter()
        .by_uniform_blocks(block)
        .map(f)
        .collect::<Result<Vec<_>, _>>()
}

fn numa_pools() -> Option<&'static [NumaPool]> {
    NUMA_POOLS.get_or_init(build_numa_pools).as_deref()
}

#[cfg(target_os = "linux")]
fn build_numa_pools() -> Option<Vec<NumaPool>> {
    if !numa_awareness_enabled() {
        return None;
    }

    let all_cores = core_affinity::get_core_ids()?;
    let mut nodes = linux_numa_node_cores(&all_cores)?;
    if nodes.len() < 2 {
        return None;
    }

    let total_threads = rayon_thread_count(all_cores.len());
    if total_threads < 2 {
        return None;
    }

    let node_count = nodes.len();
    let base = total_threads / node_count;
    let remainder = total_threads % node_count;

    let mut pools = Vec::with_capacity(node_count);
    for (idx, node_cores) in nodes.iter_mut().enumerate() {
        let requested = base + usize::from(idx < remainder);
        let threads = requested.min(node_cores.len());
        if threads == 0 {
            continue;
        }

        let pinned_cores = Arc::new(node_cores.clone());
        let start_handler_cores = Arc::clone(&pinned_cores);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .start_handler(move |worker_idx| {
                if let Some(core_id) =
                    start_handler_cores.get(worker_idx % start_handler_cores.len())
                {
                    let _ = core_affinity::set_for_current(*core_id);
                }
            })
            .build()
            .ok()?;
        pools.push(NumaPool { pool });
    }

    if pools.len() < 2 { None } else { Some(pools) }
}

#[cfg(not(target_os = "linux"))]
fn build_numa_pools() -> Option<Vec<NumaPool>> {
    None
}

fn static_block_size_with_threads(len: usize, threads: usize) -> usize {
    if len == 0 {
        return 1;
    }
    let threads = threads.max(1);
    let workers = len.min(threads);
    len.div_ceil(workers)
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
    let mut per_node = linux_numa_node_cores(available)?;
    if per_node.len() < 2 {
        return None;
    }

    let mut interleaved = Vec::with_capacity(available.len());
    let mut offsets = vec![0usize; per_node.len()];

    loop {
        let mut pushed = false;
        for (node_idx, node_cpus) in per_node.iter().enumerate() {
            if let Some(core_id) = node_cpus.get(offsets[node_idx]) {
                interleaved.push(*core_id);
                offsets[node_idx] += 1;
                pushed = true;
            }
        }
        if !pushed {
            break;
        }
    }

    Some(interleaved)
}

#[cfg(target_os = "linux")]
fn linux_numa_node_cores(available: &[CoreId]) -> Option<Vec<Vec<CoreId>>> {
    let available_map: HashMap<usize, CoreId> = available.iter().map(|c| (c.id, *c)).collect();
    let available_set: HashSet<usize> = available_map.keys().copied().collect();

    let mut per_node: Vec<Vec<CoreId>> = std::fs::read_dir("/sys/devices/system/node")
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
            let mut cpus: Vec<CoreId> = parse_linux_cpu_list(&cpulist)?
                .into_iter()
                .filter(|cpu| available_set.contains(cpu))
                .filter_map(|cpu| available_map.get(&cpu).copied())
                .collect();
            cpus.sort_unstable_by_key(|c| c.id);
            if cpus.is_empty() { None } else { Some(cpus) }
        })
        .collect();

    if per_node.is_empty() {
        return None;
    }
    per_node.sort_by_key(|cpus| cpus[0].id);
    Some(per_node)
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
    #[test]
    fn static_block_size_balances_even_work() {
        assert_eq!(super::static_block_size_with_threads(120, 12), 10);
    }

    #[test]
    fn static_block_size_balances_uneven_work() {
        assert_eq!(super::static_block_size_with_threads(121, 12), 11);
    }

    #[test]
    fn static_block_size_avoids_zero() {
        assert_eq!(super::static_block_size_with_threads(0, 12), 1);
        assert_eq!(super::static_block_size_with_threads(1, 144), 1);
    }

    #[test]
    fn static_block_size_with_threads_cap() {
        assert_eq!(super::static_block_size_with_threads(12, 144), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_cpu_list() {
        let parsed = super::parse_linux_cpu_list("0-2,5,8-9").unwrap();
        assert_eq!(parsed, vec![0, 1, 2, 5, 8, 9]);
    }
}
