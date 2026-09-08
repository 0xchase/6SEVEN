#ifndef SIXSEVEN_PLUGIN_H
#define SIXSEVEN_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t *data;
    size_t len;
} SixsevenBuffer;

typedef struct {
    uint32_t abi_version;
    size_t struct_size;
    void *(*create)(void);
    int32_t (*call)(void *, const uint8_t *, size_t, SixsevenBuffer *);
    void (*release)(void *, SixsevenBuffer);
    void (*destroy)(void *);
} SixsevenPluginApi;

const SixsevenPluginApi *sixseven_plugin_v1(void);

#ifdef __cplusplus
}
#endif

#endif
