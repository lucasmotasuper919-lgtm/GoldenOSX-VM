#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "lvp_private.h"
#include "nir.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "metal_compiler.h"

/* Extracts Lavapipe's per-stage nir_shader_compiler_options from a VkPhysicalDevice
 * handle created by our matching Mesa 26.2.5 libvulkan_lvp.so. */
void stela_lvp_nir_options(void *vk_physical_device, const void *out[8]) {
    struct lvp_physical_device *pdev = (struct lvp_physical_device *)vk_physical_device;
    const nir_shader_compiler_options *fallback = pdev->drv_options[MESA_SHADER_COMPUTE];
    if (!fallback) {
        fallback = pdev->drv_options[MESA_SHADER_FRAGMENT];
    }
    for (int i = 0; i < 8; i++) {
        const nir_shader_compiler_options *opt = NULL;
        if (i < LVP_SHADER_STAGES) {
            opt = pdev->drv_options[i];
        }
        out[i] = opt ? (const void *)opt : (const void *)fallback;
    }
}

/* Generic Vulkan-compatible NIR compiler options for external hardware Vulkan ICDs
 * (Intel ANV / AMD RADV / NVIDIA NVK) when not using Lavapipe's internal struct. */
static const nir_shader_compiler_options generic_hw_nir_options = {
    .lower_scmp = true,
    .lower_flrp32 = true,
    .lower_flrp64 = true,
    .lower_fsat = true,
    .lower_bitfield_insert = true,
    .lower_bitfield_extract = true,
    .lower_fdiv = false,
    .max_unroll_iterations = 32,
};

void stela_generic_nir_options(const void *out[8]) {
    for (int i = 0; i < 8; i++) {
        out[i] = (const void *)&generic_hw_nir_options;
    }
}

/* 128-bit FNV-1a streaming hasher (two independent 64-bit lanes) for AIR->NIR cache keys. */
struct stela_hash128 {
    uint64_t h0;
    uint64_t h1;
};

static inline void hash_init(struct stela_hash128 *h) {
    h->h0 = 0xcbf29ce484222325ULL;
    h->h1 = 0x84222325cbf29ce4ULL;
}

static inline void hash_bytes(struct stela_hash128 *h, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h0 = h->h0;
    uint64_t h1 = h->h1;
    for (size_t i = 0; i < len; i++) {
        h0 ^= (uint64_t)p[i];
        h0 *= 0x100000001b3ULL;
        h1 ^= (uint64_t)(p[i] ^ (uint8_t)(i & 0xff));
        h1 *= 0x100000001b3ULL;
    }
    h->h0 = h0;
    h->h1 = h1;
}

#define HASH_VAL(h, v) hash_bytes((h), &(v), sizeof(v))

#define STELA_NIR_CACHE_MAGIC 0x52494E53u /* "SNIR" */
#define STELA_NIR_CACHE_VER   1u

struct stela_nir_cache_hdr {
    uint32_t magic;
    uint32_t version;
    uint32_t mesa_stage;
    uint32_t data_bytes;
};

/* Disk-cached wrapper around stela_metal_compile_reflected().
 * Serializes the compiled nir_shader via Mesa's nir_serialize() / nir_deserialize(),
 * turning subsequent boots' 225+ Metal shader compilations from ~200-500ms LLVM runs
 * into <0.5ms cache hits. Sets *out_hit = 1 on cache hit, 0 on miss. */
void *stela_metal_compile_cached(
    const char *cache_dir,
    const void *air,
    size_t air_size,
    const char *entry,
    const struct stela_metal_target *target,
    const struct stela_metal_specialization *spec,
    const struct stela_metal_reflection *reflected,
    struct stela_air_error *error,
    int *out_hit
) {
    if (out_hit) *out_hit = 0;

    bool cacheable = (cache_dir != NULL && cache_dir[0] != '\0' &&
                      air != NULL && air_size > 0 && entry != NULL &&
                      target != NULL && reflected != NULL);
    if (cacheable && spec != NULL) {
        if (spec->module_count > 0 || spec->linked_function_count > 0 ||
            spec->tessellation != NULL) {
            cacheable = false;
        }
    }

    if (!cacheable) {
        return stela_metal_compile_reflected(air, air_size, entry, target, spec, reflected, error);
    }

    struct stela_hash128 h;
    hash_init(&h);
    const char tag[] = "stela-nir-v1";
    hash_bytes(&h, tag, sizeof(tag));
    hash_bytes(&h, air, air_size);
    hash_bytes(&h, entry, strlen(entry) + 1);

    /* Hash scalar Target fields (skip nir pointer array). */
    HASH_VAL(&h, target->max_instructions);
    HASH_VAL(&h, target->max_threadgroup_bytes);
    HASH_VAL(&h, target->max_private_bytes);
    HASH_VAL(&h, target->max_constant_bytes);
    HASH_VAL(&h, target->max_varying_slots);
    HASH_VAL(&h, target->max_payload_bytes);
    HASH_VAL(&h, target->max_mesh_vertices);
    HASH_VAL(&h, target->max_mesh_primitives);
    HASH_VAL(&h, target->stencil_export);
    HASH_VAL(&h, target->raster_order_groups);
    HASH_VAL(&h, target->fragment_tests);

    HASH_VAL(&h, reflected->stage);

    uint32_t has_spec = spec ? 1u : 0u;
    HASH_VAL(&h, has_spec);
    if (spec) {
        HASH_VAL(&h, spec->workgroup_size);
        HASH_VAL(&h, spec->threadgroup_length_count);
        if (spec->threadgroup_length_count > 0 && spec->threadgroup_lengths) {
            hash_bytes(&h, spec->threadgroup_lengths,
                       spec->threadgroup_length_count * sizeof(*spec->threadgroup_lengths));
        }
        HASH_VAL(&h, spec->varying_count);
        if (spec->varying_count > 0 && spec->varyings) {
            for (uint32_t i = 0; i < spec->varying_count; i++) {
                HASH_VAL(&h, spec->varyings[i].identifier_bytes);
                if (spec->varyings[i].identifier && spec->varyings[i].identifier_bytes > 0) {
                    hash_bytes(&h, spec->varyings[i].identifier, spec->varyings[i].identifier_bytes);
                }
                HASH_VAL(&h, spec->varyings[i].per_primitive);
            }
        }
        HASH_VAL(&h, spec->amplified);
        HASH_VAL(&h, spec->launched_by_grid);
        HASH_VAL(&h, spec->dynamic_threadgroups);
        HASH_VAL(&h, spec->constant_count);
        if (spec->constant_count > 0 && spec->constants) {
            for (uint32_t i = 0; i < spec->constant_count; i++) {
                HASH_VAL(&h, spec->constants[i].index);
                HASH_VAL(&h, spec->constants[i].size);
                if (spec->constants[i].bytes && spec->constants[i].size > 0) {
                    hash_bytes(&h, spec->constants[i].bytes, spec->constants[i].size);
                }
            }
        }
        uint32_t has_stage_in = spec->stage_in ? 1u : 0u;
        HASH_VAL(&h, has_stage_in);
        if (spec->stage_in) {
            hash_bytes(&h, spec->stage_in, STELA_METAL_STAGE_IN_BYTES);
        }
        HASH_VAL(&h, spec->max_call_stack_depth);
        HASH_VAL(&h, spec->sampler_count);
        if (spec->sampler_count > 0 && spec->samplers) {
            hash_bytes(&h, spec->samplers,
                       spec->sampler_count * STELA_METAL_CONSTEXPR_SAMPLER_BYTES);
        }
        HASH_VAL(&h, spec->attachment_count);
        if (spec->attachment_count > 0 && spec->attachments) {
            hash_bytes(&h, spec->attachments,
                       spec->attachment_count * sizeof(*spec->attachments));
        }
        HASH_VAL(&h, spec->samples);
        HASH_VAL(&h, spec->round_toward_zero);
        HASH_VAL(&h, spec->extended_range);
        HASH_VAL(&h, spec->vertex_formats);
        HASH_VAL(&h, spec->imageblock_size);
        HASH_VAL(&h, spec->tile);
        HASH_VAL(&h, spec->fragment_tests);
        HASH_VAL(&h, spec->bicubic);
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/%016llx%016llx.nir",
             cache_dir, (unsigned long long)h.h0, (unsigned long long)h.h1);

    /* Try cache read */
    FILE *fp = fopen(path, "rb");
    if (fp) {
        struct stela_nir_cache_hdr hdr;
        if (fread(&hdr, sizeof(hdr), 1, fp) == 1 &&
            hdr.magic == STELA_NIR_CACHE_MAGIC &&
            hdr.version == STELA_NIR_CACHE_VER &&
            hdr.mesa_stage < STELA_METAL_NIR_STAGES &&
            hdr.data_bytes > 0 && hdr.data_bytes <= 16u * 1024u * 1024u &&
            target->nir[hdr.mesa_stage] != NULL) {
            uint8_t *buf = (uint8_t *)malloc(hdr.data_bytes);
            if (buf) {
                if (fread(buf, 1, hdr.data_bytes, fp) == hdr.data_bytes) {
                    struct blob_reader reader;
                    blob_reader_init(&reader, buf, hdr.data_bytes);
                    nir_shader *nir = nir_deserialize(NULL, target->nir[hdr.mesa_stage], &reader);
                    if (nir && !reader.overrun) {
                        free(buf);
                        fclose(fp);
                        if (out_hit) *out_hit = 1;
                        return (void *)nir;
                    }
                    if (nir) {
                        ralloc_free(nir);
                    }
                }
                free(buf);
            }
        }
        fclose(fp);
    }

    /* Cache miss: compile via LLVM AIR->NIR */
    nir_shader *nir = (nir_shader *)stela_metal_compile_reflected(
        air, air_size, entry, target, spec, reflected, error);
    if (!nir) {
        return NULL;
    }

    /* Serialize to disk atomically */
    if ((uint32_t)nir->info.stage < STELA_METAL_NIR_STAGES) {
        struct blob blob;
        blob_init(&blob);
        nir_serialize(&blob, nir, false);
        if (!blob.out_of_memory && blob.size > 0) {
            char tmp[544];
            snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
            FILE *wfp = fopen(tmp, "wb");
            if (wfp) {
                struct stela_nir_cache_hdr hdr = {
                    .magic = STELA_NIR_CACHE_MAGIC,
                    .version = STELA_NIR_CACHE_VER,
                    .mesa_stage = (uint32_t)nir->info.stage,
                    .data_bytes = (uint32_t)blob.size,
                };
                if (fwrite(&hdr, sizeof(hdr), 1, wfp) == 1 &&
                    fwrite(blob.data, 1, blob.size, wfp) == blob.size) {
                    fclose(wfp);
                    rename(tmp, path);
                } else {
                    fclose(wfp);
                    unlink(tmp);
                }
            }
        }
        blob_finish(&blob);
    }

    return (void *)nir;
}
