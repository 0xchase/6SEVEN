pub(super) fn numpy_argsort(values: &[usize]) -> Vec<usize> {
    let n = values.len();
    let mut tosort: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return tosort;
    }
    numpy_aquicksort(values, &mut tosort);
    tosort
}

const SMALL_QUICKSORT: usize = 15;

fn numpy_aquicksort(v: &[usize], tosort: &mut [usize]) {
    let n = tosort.len();
    if n <= 1 {
        return;
    }

    // npy_get_msb returns floor(log2(n)). initial depth = 2 * floor(log2(n))
    let msb = (usize::BITS - n.leading_zeros() - 1) as i32;
    let mut cdepth: i32 = msb * 2;

    // Stack stores (left, right, depth) for deferred partitions
    let mut stack: Vec<(usize, usize, i32)> = Vec::new();
    let mut pl: usize = 0;
    let mut pr: usize = n - 1;

    loop {
        if cdepth < 0 {
            // Heapsort fallback for this subarray
            numpy_aheapsort(v, &mut tosort[pl..=pr]);
            if let Some((l, r, d)) = stack.pop() {
                pl = l;
                pr = r;
                cdepth = d;
            } else {
                break;
            }
            continue;
        }

        while pr.wrapping_sub(pl) > SMALL_QUICKSORT {
            // Quicksort partition with median-of-3 pivot
            let pm = pl + ((pr - pl) >> 1);
            if v[tosort[pm]] < v[tosort[pl]] {
                tosort.swap(pm, pl);
            }
            if v[tosort[pr]] < v[tosort[pm]] {
                tosort.swap(pr, pm);
            }
            if v[tosort[pm]] < v[tosort[pl]] {
                tosort.swap(pm, pl);
            }
            let vp = v[tosort[pm]];

            let mut pi = pl;
            let mut pj = pr - 1;
            tosort.swap(pm, pj);

            loop {
                pi += 1;
                while v[tosort[pi]] < vp {
                    pi += 1;
                }
                pj -= 1;
                while vp < v[tosort[pj]] {
                    pj -= 1;
                }
                if pi >= pj {
                    break;
                }
                tosort.swap(pi, pj);
            }

            tosort.swap(pi, pr - 1);

            // Push largest partition on stack, continue with smallest
            cdepth -= 1;
            if pi - pl < pr - pi {
                stack.push((pi + 1, pr, cdepth));
                pr = pi.wrapping_sub(1); // may wrap if pi==0, caught by while condition
            } else {
                stack.push((pl, pi.wrapping_sub(1), cdepth));
                pl = pi + 1;
            }
        }

        // Insertion sort for small partition [pl..=pr]
        if pr > pl {
            let mut i = pl + 1;
            while i <= pr {
                let vi = tosort[i];
                let vp = v[vi];
                let mut j = i;
                while j > pl && vp < v[tosort[j - 1]] {
                    tosort[j] = tosort[j - 1];
                    j -= 1;
                }
                tosort[j] = vi;
                i += 1;
            }
        }

        // Pop stack
        if let Some((l, r, d)) = stack.pop() {
            pl = l;
            pr = r;
            cdepth = d;
        } else {
            break;
        }
    }
}

fn numpy_aheapsort(v: &[usize], tosort: &mut [usize]) {
    let mut nn = tosort.len();
    if nn <= 1 {
        return;
    }

    // Build max-heap (1-indexed: a[i] = tosort[i-1])
    let mut l = nn >> 1;
    while l > 0 {
        let tmp = tosort[l - 1];
        let mut i = l;
        let mut j = l << 1;
        while j <= nn {
            if j < nn && v[tosort[j - 1]] < v[tosort[j]] {
                j += 1;
            }
            if v[tmp] < v[tosort[j - 1]] {
                tosort[i - 1] = tosort[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
        l -= 1;
    }

    // Extract sorted
    while nn > 1 {
        let tmp = tosort[nn - 1];
        tosort[nn - 1] = tosort[0];
        nn -= 1;
        let mut i = 1;
        let mut j = 2;
        while j <= nn {
            if j < nn && v[tosort[j - 1]] < v[tosort[j]] {
                j += 1;
            }
            if v[tmp] < v[tosort[j - 1]] {
                tosort[i - 1] = tosort[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
    }
}
