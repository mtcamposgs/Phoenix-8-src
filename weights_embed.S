#if (defined(_WIN32) || defined(__CYGWIN__)) && !defined(__x86_64__) && !defined(_M_X64)
#define SYM(name) _##name
#elif defined(__APPLE__)
#define SYM(name) _##name
#else
#define SYM(name) name
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
.section .rdata,"dr"
#elif defined(__APPLE__)
.section __TEXT,__const
#else
.section .rodata
#endif

.balign 64
.globl SYM(g_embedded_shra_weights_start)
SYM(g_embedded_shra_weights_start):
.incbin "shra_weights.bin"
.globl SYM(g_embedded_shra_weights_end)
SYM(g_embedded_shra_weights_end):