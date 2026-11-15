#include "defs.h"
#include <immintrin.h>

typedef struct {
    uint64_t key;
    uint64_t data;
} TT_Entry;

typedef struct {
    TT_Entry entry[4];
} TT_Bucket;

static TT_Bucket* transposition_table = NULL;
static size_t g_tt_buckets = 0;
static uint8_t g_tt_current_age = 0;

static inline size_t tt_index(U64 hash) {
    return (size_t)(((uint64_t)(uint32_t)(hash >> 32) * (uint64_t)g_tt_buckets) >> 32);
}

static inline uint64_t pack_tt_data(int32_t score, uint16_t move16, int depth, int flags, uint8_t age) {
    uint8_t d = (depth < 0) ? 0 : ((depth > 255) ? 255 : (uint8_t)depth);
    uint8_t fa = (uint8_t)((flags & 0x3) | ((age & 0x3F) << 2));
    return (uint64_t)(uint32_t)score
         | ((uint64_t)(move16 & 0x7FFFU) << 32)
         | ((uint64_t)d << 48)
         | ((uint64_t)fa << 56);
}

void init_tt(int size_mb) {
    if (size_mb < 1) size_mb = 1;
    if (size_mb > 16384) size_mb = 16384;

    if (transposition_table) _mm_free(transposition_table);

    size_t bytes = (size_t)size_mb * 1024ULL * 1024ULL;
    g_tt_buckets = bytes / sizeof(TT_Bucket);

    transposition_table = (TT_Bucket*)_mm_malloc(g_tt_buckets * sizeof(TT_Bucket), 64);
    if (!transposition_table) {
        fprintf(stderr, "Failed to allocate transposition table (%d MB)\n", size_mb);
        exit(EXIT_FAILURE);
    }
    g_tt_current_age = 0;
    clear_tt();
}

void clear_tt(void) {
    if (transposition_table && g_tt_buckets > 0) {
        memset(transposition_table, 0, g_tt_buckets * sizeof(TT_Bucket));
    }
}

void new_search_tt(void) {
    g_tt_current_age = (uint8_t)((g_tt_current_age + 1) & 0x3F);
}

int get_hashfull(void) {
    if (!transposition_table || g_tt_buckets == 0) return 0;
    int count = 0;
    int sample_size = (g_tt_buckets < 1000) ? (int)g_tt_buckets : 1000;
    uint8_t cur_age = g_tt_current_age & 0x3F;
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < 4; ++j) {
            uint64_t k = transposition_table[i].entry[j].key;
            uint64_t d = transposition_table[i].entry[j].data;
            if (k != 0 || d != 0) {
                uint8_t age = (uint8_t)((d >> 58) & 0x3F);
                if (age == cur_age) count++;
            }
        }
    }
    return (count * 1000) / (sample_size * 4);
}

bool probe_tt_full(U64 hash, int depth, int alpha, int beta, int ply,
                   int* out_score, Move* out_tt_move,
                   int* out_tt_depth, int* out_tt_flags, bool* out_cutoff) {
    *out_tt_move = MOVE_NONE;
    *out_score = 0;
    *out_tt_depth = -1;
    *out_tt_flags = 0;
    *out_cutoff = false;

    if (!transposition_table) return false;

    TT_Bucket* bucket = &transposition_table[tt_index(hash)];

    for (int i = 0; i < 4; ++i) {
        uint64_t d = bucket->entry[i].data;
        uint64_t k = bucket->entry[i].key;
        if ((k ^ d) == hash) {
            uint16_t move16 = (uint16_t)((d >> 32) & 0x7FFFU);
            int e_depth = (int)((d >> 48) & 0xFF);
            int e_flags = (int)((d >> 56) & 0x3);
            int score = (int)(int32_t)(uint32_t)(d & 0xFFFFFFFFULL);

            if (score > MATE_SCORE - MAX_PLY) score -= ply;
            if (score < -MATE_SCORE + MAX_PLY) score += ply;

            *out_tt_move = (Move)move16;
            *out_tt_depth = e_depth;
            *out_tt_flags = e_flags;
            *out_score = score;

            if (e_depth >= depth) {
                if (e_flags == TT_EXACT) {
                    *out_cutoff = true;
                } else if (e_flags == TT_ALPHA && score <= alpha) {
                    *out_score = alpha;
                    *out_cutoff = true;
                } else if (e_flags == TT_BETA && score >= beta) {
                    *out_score = beta;
                    *out_cutoff = true;
                }
            }
            return true;
        }
    }
    return false;
}

void store_tt(U64 hash, int depth, int flags, int score, Move best, int ply) {
    if (!transposition_table) return;

    if (score > MATE_SCORE - MAX_PLY) score += ply;
    if (score < -MATE_SCORE + MAX_PLY) score -= ply;

    uint16_t move16 = (uint16_t)(best & 0x7FFFU);
    TT_Bucket* bucket = &transposition_table[tt_index(hash)];
    uint8_t cur_age = g_tt_current_age & 0x3F;

    int victim_idx = 0;
    int worst_value = 1000000;

    for (int i = 0; i < 4; ++i) {
        uint64_t d = bucket->entry[i].data;
        uint64_t k = bucket->entry[i].key;

        if (k == 0 && d == 0) {
            victim_idx = i;
            break;
        }

        if ((k ^ d) == hash) {
            int old_depth = (int)((d >> 48) & 0xFF);
            uint8_t old_age = (uint8_t)((d >> 58) & 0x3F);
            if (move16 == 0) {
                move16 = (uint16_t)((d >> 32) & 0x7FFFU);
            }
            if (flags == TT_EXACT || depth + 2 >= old_depth || old_age != cur_age) {
                uint64_t new_data = pack_tt_data((int32_t)score, move16, depth, flags, cur_age);
                bucket->entry[i].data = new_data;
                bucket->entry[i].key = hash ^ new_data;
            }
            return;
        }

        int old_depth = (int)((d >> 48) & 0xFF);
        uint8_t old_age = (uint8_t)((d >> 58) & 0x3F);
        int age_dist = (int)((cur_age - old_age) & 0x3F);
        int val = old_depth - age_dist * 4;
        if (val < worst_value) {
            worst_value = val;
            victim_idx = i;
        }
    }

    uint64_t new_data = pack_tt_data((int32_t)score, move16, depth, flags, cur_age);
    bucket->entry[victim_idx].data = new_data;
    bucket->entry[victim_idx].key = hash ^ new_data;
}