asm volatile(
    "%[i] &= %[mask]"
    : [i] "+r" (i)
    : [mask] "i" (MAX_CANDIDATES - 1)
);


    #define EVICT_LOOP_BODY(i) \
	do { \
		if ((i) < num_to_evict && (i) < 32) { \
			__u64 fkey = (*candidates)[(i)].folio_addr; \
			eviction_ctx->folios_to_evict[(i)] = (struct folio *)fkey; \
			eviction_ctx->nr_folios_to_evict++; \
		} \
	} while (0)