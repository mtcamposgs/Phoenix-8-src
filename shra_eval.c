#include "defs.h"
#include <immintrin.h>

extern const uint8_t g_embedded_shra_weights_start[];
extern const uint8_t g_embedded_shra_weights_end[];

const int piece_values[PIECE_COUNT] = {0, 100, 320, 330, 500, 975, 10000, 0, 0, 0, 0, 0, 0, 0};

static const int KING_BUCKET_MAP[32] = {
    0,  1,  2,  3,
    4,  4,  5,  5,
    6,  6,  7,  7,
    6,  6,  7,  7,
    8,  8,  9,  9,
    8,  8,  9,  9,
   10, 10, 11, 11,
   10, 10, 11, 11
};

static int16_t* g_w_spatial    = NULL; // [9222 * 1024]
static int16_t* g_b_spatial    = NULL; // [1024]
static int32_t* g_psqt         = NULL; // [9216 * 8]
static int16_t* g_w_line_ortho = NULL; // [8]
static int16_t* g_w_line_diag  = NULL; // [8]
static int16_t* g_w_geo        = NULL; // [512]
static int16_t* g_w_tac        = NULL; // [512]
static int16_t* g_chan_w       = NULL; // [8 * 8]
static int32_t* g_chan_b       = NULL; // [8]
static int8_t*  g_fc1_w        = NULL; // [32 * 2048]
static int32_t* g_fc1_b        = NULL; // [32]
static int16_t* g_fc2_w        = NULL; // [8 * 32 * 32]
static int32_t* g_fc2_b        = NULL; // [8 * 32]
static int16_t* g_fc3_w        = NULL; // [8 * 32]
static int32_t* g_fc3_b        = NULL; // [8]

static size_t shra_expected_bytes(void) {
    return (size_t)SHRA_TOTAL_INPUTS * SHRA_D * sizeof(int16_t)
         + (size_t)SHRA_D * sizeof(int16_t)
         + (size_t)SHRA_NUM_KING_FEATS * SHRA_PHASE_BUCKETS * sizeof(int32_t)
         + (size_t)SHRA_CHANNELS * sizeof(int16_t)
         + (size_t)SHRA_CHANNELS * sizeof(int16_t)
         + (size_t)SHRA_SPATIAL_D * sizeof(int16_t)
         + (size_t)SHRA_SPATIAL_D * sizeof(int16_t)
         + (size_t)SHRA_CHANNELS * SHRA_CHANNELS * sizeof(int16_t)
         + (size_t)SHRA_CHANNELS * sizeof(int32_t)
         + (size_t)SHRA_FC1_DIM * (2 * SHRA_D) * sizeof(int8_t)
         + (size_t)SHRA_FC1_DIM * sizeof(int32_t)
         + (size_t)SHRA_PHASE_BUCKETS * SHRA_FC2_DIM * SHRA_FC1_DIM * sizeof(int16_t)
         + (size_t)SHRA_PHASE_BUCKETS * SHRA_FC2_DIM * sizeof(int32_t)
         + (size_t)SHRA_PHASE_BUCKETS * SHRA_FC2_DIM * sizeof(int16_t)
         + (size_t)SHRA_PHASE_BUCKETS * sizeof(int32_t);
}

static bool allocate_shra_buffers(void) {
    if (g_w_spatial) return true;

    g_w_spatial    = (int16_t*)_mm_malloc(SHRA_TOTAL_INPUTS * SHRA_D * sizeof(int16_t), 32);
    g_b_spatial    = (int16_t*)_mm_malloc(SHRA_D * sizeof(int16_t), 32);
    g_psqt         = (int32_t*)_mm_malloc(SHRA_NUM_KING_FEATS * SHRA_PHASE_BUCKETS * sizeof(int32_t), 32);
    g_w_line_ortho = (int16_t*)_mm_malloc(SHRA_CHANNELS * sizeof(int16_t), 32);
    g_w_line_diag  = (int16_t*)_mm_malloc(SHRA_CHANNELS * sizeof(int16_t), 32);
    g_w_geo        = (int16_t*)_mm_malloc(SHRA_SPATIAL_D * sizeof(int16_t), 32);
    g_w_tac        = (int16_t*)_mm_malloc(SHRA_SPATIAL_D * sizeof(int16_t), 32);
    g_chan_w       = (int16_t*)_mm_malloc(SHRA_CHANNELS * SHRA_CHANNELS * sizeof(int16_t), 32);
    g_chan_b       = (int32_t*)_mm_malloc(SHRA_CHANNELS * sizeof(int32_t), 32);
    g_fc1_w        = (int8_t*) _mm_malloc(SHRA_FC1_DIM * (2 * SHRA_D) * sizeof(int8_t), 32);
    g_fc1_b        = (int32_t*)_mm_malloc(SHRA_FC1_DIM * sizeof(int32_t), 32);
    g_fc2_w        = (int16_t*)_mm_malloc(SHRA_PHASE_BUCKETS * SHRA_FC2_DIM * SHRA_FC1_DIM * sizeof(int16_t), 32);
    g_fc2_b        = (int32_t*)_mm_malloc(SHRA_PHASE_BUCKETS * SHRA_FC2_DIM * sizeof(int32_t), 32);
    g_fc3_w        = (int16_t*)_mm_malloc(SHRA_PHASE_BUCKETS * SHRA_FC2_DIM * sizeof(int16_t), 32);
    g_fc3_b        = (int32_t*)_mm_malloc(SHRA_PHASE_BUCKETS * sizeof(int32_t), 32);

    return g_w_spatial && g_b_spatial && g_psqt && g_w_line_ortho && g_w_line_diag &&
           g_w_geo && g_w_tac && g_chan_w && g_chan_b && g_fc1_w && g_fc1_b &&
           g_fc2_w && g_fc2_b && g_fc3_w && g_fc3_b;
}

static bool init_shra_from_memory(const uint8_t* data, size_t size) {
    if (size != shra_expected_bytes()) return false;
    if (!allocate_shra_buffers()) return false;

    const uint8_t* ptr = data;
    size_t sz;

    #define LOAD_PART(dst, count, type) do { \
        sz = (size_t)(count) * sizeof(type); \
        memcpy((dst), ptr, sz); \
        ptr += sz; \
    } while (0)

    LOAD_PART(g_w_spatial,    SHRA_TOTAL_INPUTS * SHRA_D, int16_t);
    LOAD_PART(g_b_spatial,    SHRA_D, int16_t);
    LOAD_PART(g_psqt,         SHRA_NUM_KING_FEATS * SHRA_PHASE_BUCKETS, int32_t);
    LOAD_PART(g_w_line_ortho, SHRA_CHANNELS, int16_t);
    LOAD_PART(g_w_line_diag,  SHRA_CHANNELS, int16_t);
    LOAD_PART(g_w_geo,        SHRA_SPATIAL_D, int16_t);
    LOAD_PART(g_w_tac,        SHRA_SPATIAL_D, int16_t);
    LOAD_PART(g_chan_w,       SHRA_CHANNELS * SHRA_CHANNELS, int16_t);
    LOAD_PART(g_chan_b,       SHRA_CHANNELS, int32_t);
    LOAD_PART(g_fc1_w,        SHRA_FC1_DIM * (2 * SHRA_D), int8_t);
    LOAD_PART(g_fc1_b,        SHRA_FC1_DIM, int32_t);
    LOAD_PART(g_fc2_w,        SHRA_PHASE_BUCKETS * SHRA_FC2_DIM * SHRA_FC1_DIM, int16_t);
    LOAD_PART(g_fc2_b,        SHRA_PHASE_BUCKETS * SHRA_FC2_DIM, int32_t);
    LOAD_PART(g_fc3_w,        SHRA_PHASE_BUCKETS * SHRA_FC2_DIM, int16_t);
    LOAD_PART(g_fc3_b,        SHRA_PHASE_BUCKETS, int32_t);

    #undef LOAD_PART
    return true;
}

bool init_shra_embedded(void) {
    size_t embedded_size = (size_t)(g_embedded_shra_weights_end - g_embedded_shra_weights_start);
    return init_shra_from_memory(g_embedded_shra_weights_start, embedded_size);
}

bool init_shra(const char* bin_path) {
    if (!bin_path || bin_path[0] == '\0' || strcmp(bin_path, "<internal>") == 0) {
        return init_shra_embedded();
    }

    FILE* f = fopen(bin_path, "rb");
    if (!f) return false;

    const size_t expected_bytes = shra_expected_bytes();
    fseek(f, 0, SEEK_END);
    long actual_bytes = ftell(f);
    rewind(f);

    if (actual_bytes < 0 || (size_t)actual_bytes != expected_bytes) {
        fclose(f);
        return false;
    }

    uint8_t* buf = (uint8_t*)malloc(expected_bytes);
    if (!buf) {
        fclose(f);
        return false;
    }

    size_t read_bytes = fread(buf, 1, expected_bytes, f);
    fclose(f);

    bool ok = (read_bytes == expected_bytes) && init_shra_from_memory(buf, expected_bytes);
    free(buf);
    return ok;
}

int shra_king_bucket_id(int ksq, int color) {
    int ksq_p = (color == BLACK) ? (ksq ^ 56) : ksq;
    int mirror = ((ksq_p & 7) >= 4) ? 1 : 0;
    int ksq_canon = ksq_p ^ (mirror ? 7 : 0);
    int bucket = KING_BUCKET_MAP[(ksq_canon >> 3) * 4 + (ksq_canon & 3)];
    return (bucket << 1) | mirror;
}

static inline int make_king_feat(int k_bucket, int mirror_mask, int side, int sq, int pt0, int p_color) {
    int c_sq = ((side == BLACK) ? (sq ^ 56) : sq) ^ mirror_mask;
    int rel_pt = (p_color == side) ? pt0 : (pt0 + 6);
    return k_bucket * SHRA_FEAT_PER_BUCKET + rel_pt * 64 + c_sq;
}

static inline void acc_add_i16(int16_t* dst, const int16_t* src) {
    for (int i = 0; i < SHRA_D; i += 16) {
        __m256i v = _mm256_add_epi16(_mm256_load_si256((const __m256i*)&dst[i]),
                                     _mm256_load_si256((const __m256i*)&src[i]));
        _mm256_store_si256((__m256i*)&dst[i], v);
    }
}

static inline void acc_sub_add_i16(const int16_t* in, int16_t* out, const int16_t* sub_p, const int16_t* add_p) {
    for (int i = 0; i < SHRA_D; i += 16) {
        __m256i v = _mm256_load_si256((const __m256i*)&in[i]);
        v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i*)&sub_p[i]));
        v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i*)&add_p[i]));
        _mm256_store_si256((__m256i*)&out[i], v);
    }
}

void shra_init_single_accumulator(const Board* board, int side, int16_t* out_acc, int32_t* out_psqt) {
    memcpy(out_acc, g_b_spatial, SHRA_D * sizeof(int16_t));
    __m256i psqt_v = _mm256_setzero_si256();

    int id = shra_king_bucket_id(board->king_sq[side], side);
    int k_bucket = id >> 1;
    int mirror_mask = (id & 1) ? 7 : 0;

    for (int pt = 0; pt < 6; ++pt) {
        U64 bb_w = board->pieces[pt];
        while (bb_w) {
            int sq = pop_lsb(&bb_w);
            int feat = make_king_feat(k_bucket, mirror_mask, side, sq, pt, WHITE);
            acc_add_i16(out_acc, &g_w_spatial[feat * SHRA_D]);
            psqt_v = _mm256_add_epi32(psqt_v, _mm256_load_si256((const __m256i*)&g_psqt[feat * SHRA_PHASE_BUCKETS]));
        }
        U64 bb_b = board->pieces[6 + pt];
        while (bb_b) {
            int sq = pop_lsb(&bb_b);
            int feat = make_king_feat(k_bucket, mirror_mask, side, sq, pt, BLACK);
            acc_add_i16(out_acc, &g_w_spatial[feat * SHRA_D]);
            psqt_v = _mm256_add_epi32(psqt_v, _mm256_load_si256((const __m256i*)&g_psqt[feat * SHRA_PHASE_BUCKETS]));
        }
    }
    _mm256_store_si256((__m256i*)out_psqt, psqt_v);
}

void shra_init_accumulators(const Board* board, int16_t* out_w, int16_t* out_b, int32_t* out_psqt_w, int32_t* out_psqt_b) {
    shra_init_single_accumulator(board, WHITE, out_w, out_psqt_w);
    shra_init_single_accumulator(board, BLACK, out_b, out_psqt_b);
}

void shra_update_single_move(Move move, int piece, int color, int side, int side_ksq,
                             const int16_t* in_acc, const int32_t* in_psqt,
                             int16_t* out_acc, int32_t* out_psqt) {
    int id = shra_king_bucket_id(side_ksq, side);
    int k_bucket = id >> 1;
    int mirror_mask = (id & 1) ? 7 : 0;

    int from = move_from(move);
    int to = move_to(move);
    int promo = move_promo(move);
    int cap = move_captured(move);
    bool is_ep = move_is_ep(move);
    bool is_castle = move_is_castle(move);

    int pt0 = piece - 1;
    int dest_pt0 = (promo != EMPTY) ? (promo - 1) : pt0;

    int f_sub = make_king_feat(k_bucket, mirror_mask, side, from, pt0, color);
    int f_add = make_king_feat(k_bucket, mirror_mask, side, to, dest_pt0, color);

    acc_sub_add_i16(in_acc, out_acc, &g_w_spatial[f_sub * SHRA_D], &g_w_spatial[f_add * SHRA_D]);

    __m256i ps = _mm256_load_si256((const __m256i*)in_psqt);
    ps = _mm256_sub_epi32(ps, _mm256_load_si256((const __m256i*)&g_psqt[f_sub * SHRA_PHASE_BUCKETS]));
    ps = _mm256_add_epi32(ps, _mm256_load_si256((const __m256i*)&g_psqt[f_add * SHRA_PHASE_BUCKETS]));

    if (cap != EMPTY) {
        int cap_sq = is_ep ? (to + (color == WHITE ? -8 : 8)) : to;
        int f_cap = make_king_feat(k_bucket, mirror_mask, side, cap_sq, cap - 1, 1 - color);
        const int16_t* w_cap = &g_w_spatial[f_cap * SHRA_D];
        for (int i = 0; i < SHRA_D; i += 16) {
            __m256i v = _mm256_sub_epi16(_mm256_load_si256((const __m256i*)&out_acc[i]),
                                         _mm256_load_si256((const __m256i*)&w_cap[i]));
            _mm256_store_si256((__m256i*)&out_acc[i], v);
        }
        ps = _mm256_sub_epi32(ps, _mm256_load_si256((const __m256i*)&g_psqt[f_cap * SHRA_PHASE_BUCKETS]));
    }

    if (is_castle) {
        int r_from = -1, r_to = -1;
        switch (to) {
            case G1: r_from = H1; r_to = F1; break;
            case C1: r_from = A1; r_to = D1; break;
            case G8: r_from = H8; r_to = F8; break;
            case C8: r_from = A8; r_to = D8; break;
        }
        int f_rsub = make_king_feat(k_bucket, mirror_mask, side, r_from, ROOK - 1, color);
        int f_radd = make_king_feat(k_bucket, mirror_mask, side, r_to, ROOK - 1, color);
        acc_sub_add_i16(out_acc, out_acc, &g_w_spatial[f_rsub * SHRA_D], &g_w_spatial[f_radd * SHRA_D]);
        ps = _mm256_sub_epi32(ps, _mm256_load_si256((const __m256i*)&g_psqt[f_rsub * SHRA_PHASE_BUCKETS]));
        ps = _mm256_add_epi32(ps, _mm256_load_si256((const __m256i*)&g_psqt[f_radd * SHRA_PHASE_BUCKETS]));
    }

    _mm256_store_si256((__m256i*)out_psqt, ps);
}

void shra_update_accumulators_move(Move move, int piece, int color, int w_ksq, int b_ksq,
                                   const int16_t* in_w, const int16_t* in_b,
                                   const int32_t* in_psqt_w, const int32_t* in_psqt_b,
                                   int16_t* out_w, int16_t* out_b,
                                   int32_t* out_psqt_w, int32_t* out_psqt_b) {
    shra_update_single_move(move, piece, color, WHITE, w_ksq, in_w, in_psqt_w, out_w, out_psqt_w);
    shra_update_single_move(move, piece, color, BLACK, b_ksq, in_b, in_psqt_b, out_b, out_psqt_b);
}

static inline __m128i screlu_qa128_128(__m128i x) {
    __m128i zero = _mm_setzero_si128();
    __m128i maxv = _mm_set1_epi16(SHRA_QA);
    __m128i c = _mm_min_epi16(_mm_max_epi16(x, zero), maxv);
    return _mm_srai_epi16(_mm_mullo_epi16(c, c), 7);
}

static inline int32_t hsum_epi32(__m256i v) {
    __m128i vlow  = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i vsum  = _mm_add_epi32(vlow, vhigh);
    __m128i hi64  = _mm_unpackhi_epi64(vsum, vsum);
    __m128i sum64 = _mm_add_epi32(vsum, hi64);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(1, 1, 1, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

static void eval_perspective_avx2(const Board* board, int side, const int16_t* acc, uint8_t* out_feat) {
    int16_t z_buf[SHRA_D] __attribute__((aligned(32)));
    memcpy(z_buf, acc, SHRA_D * sizeof(int16_t));

    if (side == WHITE) {
        if (board->castling_rights & WK_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 0) * SHRA_D]);
        if (board->castling_rights & WQ_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 1) * SHRA_D]);
        if (board->castling_rights & BK_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 2) * SHRA_D]);
        if (board->castling_rights & BQ_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 3) * SHRA_D]);
    } else {
        if (board->castling_rights & BK_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 0) * SHRA_D]);
        if (board->castling_rights & BQ_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 1) * SHRA_D]);
        if (board->castling_rights & WK_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 2) * SHRA_D]);
        if (board->castling_rights & WQ_CASTLE) acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 3) * SHRA_D]);
    }
    if (board->en_passant_square != NO_SQ) {
        acc_add_i16(z_buf, &g_w_spatial[(SHRA_NUM_KING_FEATS + 4) * SHRA_D]);
    }
    if (board->halfmove_clock > 0) {
        int16_t hmc_q = (board->halfmove_clock >= 100) ? 64 : (int16_t)((board->halfmove_clock * 64) / 100);
        __m256i v_hmc = _mm256_set1_epi16(hmc_q);
        const int16_t* w_hmc = &g_w_spatial[(SHRA_NUM_KING_FEATS + 5) * SHRA_D];
        for (int i = 0; i < SHRA_D; i += 16) {
            __m256i vz = _mm256_load_si256((const __m256i*)&z_buf[i]);
            __m256i vw = _mm256_load_si256((const __m256i*)&w_hmc[i]);
            __m256i add = _mm256_srai_epi16(_mm256_mullo_epi16(vw, v_hmc), 6);
            _mm256_store_si256((__m256i*)&z_buf[i], _mm256_adds_epi16(vz, add));
        }
    }

    const __m256i v256_zero = _mm256_setzero_si256();
    const __m256i v256_128  = _mm256_set1_epi16(SHRA_QA);
    const __m256i v256_127  = _mm256_set1_epi16(127);

    for (int i = 0; i < SHRA_GLOBAL_D; i += 32) {
        __m256i z0 = _mm256_load_si256((const __m256i*)&z_buf[i]);
        __m256i z1 = _mm256_load_si256((const __m256i*)&z_buf[i + 16]);

        __m256i c0 = _mm256_min_epi16(_mm256_max_epi16(z0, v256_zero), v256_128);
        __m256i c1 = _mm256_min_epi16(_mm256_max_epi16(z1, v256_zero), v256_128);

        __m256i s0 = _mm256_min_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(c0, c0), 7), v256_127);
        __m256i s1 = _mm256_min_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(c1, c1), 7), v256_127);

        __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(s0, s1), 0xD8);
        _mm256_store_si256((__m256i*)&out_feat[i], packed);
    }

    __m128i h0_pad[12][12];
    memset(h0_pad, 0, sizeof(h0_pad));

    __m128i rank_sum[8];
    __m128i file_sum[8];
    __m128i diag_sum[15];
    __m128i adiag_sum[15];
    for (int i = 0; i < 8; ++i) { rank_sum[i] = _mm_setzero_si128(); file_sum[i] = _mm_setzero_si128(); }
    for (int i = 0; i < 15; ++i) { diag_sum[i] = _mm_setzero_si128(); adiag_sum[i] = _mm_setzero_si128(); }

    const int16_t* z_sp = &z_buf[SHRA_GLOBAL_D];
    for (int r = 0; r < 8; ++r) {
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            __m128i vz = _mm_load_si128((const __m128i*)&z_sp[sq * SHRA_CHANNELS]);
            __m128i vh = screlu_qa128_128(vz);
            h0_pad[r + 2][f + 2] = vh;
            rank_sum[r]         = _mm_add_epi16(rank_sum[r], vh);
            file_sum[f]         = _mm_add_epi16(file_sum[f], vh);
            diag_sum[r - f + 7] = _mm_add_epi16(diag_sum[r - f + 7], vh);
            adiag_sum[r + f]    = _mm_add_epi16(adiag_sum[r + f], vh);
        }
    }

    __m128i hgeo_pad[10][12];
    for (int r = 0; r < 2; ++r)
        for (int f = 0; f < 12; ++f)
            hgeo_pad[r][f] = _mm_setzero_si128();
    for (int r = 2; r < 10; ++r) {
        hgeo_pad[r][0]  = _mm_setzero_si128();
        hgeo_pad[r][1]  = _mm_setzero_si128();
        hgeo_pad[r][10] = _mm_setzero_si128();
        hgeo_pad[r][11] = _mm_setzero_si128();
    }

    const __m128i v_wlo = _mm_load_si128((const __m128i*)g_w_line_ortho);
    const __m128i v_wld = _mm_load_si128((const __m128i*)g_w_line_diag);
    const __m128i v_two = _mm_set1_epi16(2);
    const __m128i v_six = _mm_set1_epi16(6);
    const __m128i v_neg128 = _mm_set1_epi16(-128);
    const __m128i v_pos128 = _mm_set1_epi16(128);

    for (int r = 0; r < 8; ++r) {
        int pr = r + 2;
        for (int f = 0; f < 8; ++f) {
            int pf = f + 2;
            int sq = r * 8 + f;
            __m128i vh0 = h0_pad[pr][pf];

            __m128i o1 = _mm_add_epi16(
                _mm_add_epi16(h0_pad[pr + 1][pf], h0_pad[pr - 1][pf]),
                _mm_add_epi16(h0_pad[pr][pf + 1], h0_pad[pr][pf - 1]));
            __m128i o2 = _mm_add_epi16(
                _mm_add_epi16(h0_pad[pr + 2][pf], h0_pad[pr - 2][pf]),
                _mm_add_epi16(h0_pad[pr][pf + 2], h0_pad[pr][pf - 2]));

            __m128i dfwd = _mm_add_epi16(h0_pad[pr - 1][pf - 1], h0_pad[pr - 1][pf + 1]);
            __m128i dbwd = _mm_add_epi16(h0_pad[pr + 1][pf - 1], h0_pad[pr + 1][pf + 1]);
            __m128i d2 = _mm_add_epi16(
                _mm_add_epi16(h0_pad[pr + 2][pf + 2], h0_pad[pr + 2][pf - 2]),
                _mm_add_epi16(h0_pad[pr - 2][pf + 2], h0_pad[pr - 2][pf - 2]));

            __m128i kn1 = _mm_add_epi16(
                _mm_add_epi16(h0_pad[pr + 2][pf + 1], h0_pad[pr + 2][pf - 1]),
                _mm_add_epi16(h0_pad[pr - 2][pf + 1], h0_pad[pr - 2][pf - 1]));
            __m128i kn2 = _mm_add_epi16(
                _mm_add_epi16(h0_pad[pr + 1][pf + 2], h0_pad[pr + 1][pf - 2]),
                _mm_add_epi16(h0_pad[pr - 1][pf + 2], h0_pad[pr - 1][pf - 2]));

            __m128i sum_w2 = _mm_mullo_epi16(_mm_add_epi16(o1, dbwd), v_two);
            __m128i sum_w6 = _mm_mullo_epi16(dfwd, v_six);
            __m128i sum_w1 = _mm_add_epi16(o2, _mm_add_epi16(d2, _mm_add_epi16(kn1, kn2)));
            __m128i local_ray = _mm_srai_epi16(_mm_add_epi16(sum_w1, _mm_add_epi16(sum_w2, sum_w6)), 3);

            __m128i vh2 = _mm_slli_epi16(vh0, 1);
            __m128i lo = _mm_srai_epi16(_mm_sub_epi16(_mm_add_epi16(rank_sum[r], file_sum[f]), vh2), 3);
            __m128i ld = _mm_srai_epi16(_mm_sub_epi16(_mm_add_epi16(diag_sum[r - f + 7], adiag_sum[r + f]), vh2), 3);
            __m128i lray = _mm_add_epi16(
                _mm_srai_epi16(_mm_mullo_epi16(lo, v_wlo), 6),
                _mm_srai_epi16(_mm_mullo_epi16(ld, v_wld), 6));

            __m128i geo_act = screlu_qa128_128(_mm_adds_epi16(local_ray, lray));
            __m128i wgeo = _mm_load_si128((const __m128i*)&g_w_geo[sq * SHRA_CHANNELS]);
            __m128i hgeo = _mm_adds_epi16(vh0, _mm_srai_epi16(_mm_mullo_epi16(wgeo, geo_act), 6));
            hgeo = _mm_min_epi16(_mm_max_epi16(hgeo, v_neg128), v_pos128);
            hgeo_pad[pr][pf] = hgeo;
        }
    }

    const __m128i cb0 = _mm_load_si128((const __m128i*)&g_chan_b[0]);
    const __m128i cb1 = _mm_load_si128((const __m128i*)&g_chan_b[4]);
    const __m128i cw0 = _mm_load_si128((const __m128i*)&g_chan_w[0 * 8]);
    const __m128i cw1 = _mm_load_si128((const __m128i*)&g_chan_w[1 * 8]);
    const __m128i cw2 = _mm_load_si128((const __m128i*)&g_chan_w[2 * 8]);
    const __m128i cw3 = _mm_load_si128((const __m128i*)&g_chan_w[3 * 8]);
    const __m128i cw4 = _mm_load_si128((const __m128i*)&g_chan_w[4 * 8]);
    const __m128i cw5 = _mm_load_si128((const __m128i*)&g_chan_w[5 * 8]);
    const __m128i cw6 = _mm_load_si128((const __m128i*)&g_chan_w[6 * 8]);
    const __m128i cw7 = _mm_load_si128((const __m128i*)&g_chan_w[7 * 8]);
    const __m128i v_zero = _mm_setzero_si128();
    const __m128i v_pos127 = _mm_set1_epi16(127);

    __m128i hf_prev = _mm_setzero_si128();
    for (int r = 0; r < 8; ++r) {
        int pr = r + 2;
        for (int f = 0; f < 8; ++f) {
            int pf = f + 2;
            int sq = r * 8 + f;
            __m128i hg = hgeo_pad[pr][pf];

            __m128i ts1 = _mm_add_epi16(hgeo_pad[pr - 1][pf - 1], hgeo_pad[pr - 1][pf + 1]);
            __m128i ts2 = _mm_add_epi16(hgeo_pad[pr - 2][pf - 1], hgeo_pad[pr - 2][pf + 1]);
            __m128i tac_shift = _mm_srai_epi16(_mm_add_epi16(ts1, ts2), 2);
            __m128i tac_2d = screlu_qa128_128(_mm_srai_epi16(_mm_mullo_epi16(hg, tac_shift), 7));

            __m128i p0 = _mm_madd_epi16(hg, cw0);
            __m128i p1 = _mm_madd_epi16(hg, cw1);
            __m128i p2 = _mm_madd_epi16(hg, cw2);
            __m128i p3 = _mm_madd_epi16(hg, cw3);
            __m128i s01 = _mm_hadd_epi32(p0, p1);
            __m128i s23 = _mm_hadd_epi32(p2, p3);
            __m128i sum03 = _mm_srai_epi32(_mm_add_epi32(_mm_hadd_epi32(s01, s23), cb0), 6);

            __m128i p4 = _mm_madd_epi16(hg, cw4);
            __m128i p5 = _mm_madd_epi16(hg, cw5);
            __m128i p6 = _mm_madd_epi16(hg, cw6);
            __m128i p7 = _mm_madd_epi16(hg, cw7);
            __m128i s45 = _mm_hadd_epi32(p4, p5);
            __m128i s67 = _mm_hadd_epi32(p6, p7);
            __m128i sum47 = _mm_srai_epi32(_mm_add_epi32(_mm_hadd_epi32(s45, s67), cb1), 6);

            __m128i chan_mix = _mm_packs_epi32(sum03, sum47);

            __m128i tac_gate = screlu_qa128_128(_mm_adds_epi16(tac_2d, chan_mix));
            __m128i wtac = _mm_load_si128((const __m128i*)&g_w_tac[sq * SHRA_CHANNELS]);
            __m128i hfinal = _mm_adds_epi16(hg, _mm_srai_epi16(_mm_mullo_epi16(wtac, tac_gate), 6));
            hfinal = _mm_min_epi16(_mm_max_epi16(hfinal, v_zero), v_pos127);

            if ((f & 1) == 0) {
                hf_prev = hfinal;
            } else {
                __m128i packed2 = _mm_packus_epi16(hf_prev, hfinal);
                _mm_store_si128((__m128i*)&out_feat[SHRA_GLOBAL_D + (sq - 1) * SHRA_CHANNELS], packed2);
            }
        }
    }
}

int shra_evaluate_board(const Board* board,
                        const int16_t* w_acc, const int16_t* b_acc,
                        const int32_t* w_psqt, const int32_t* b_psqt,
                        ShraEvalCacheEntry* eval_cache) {
    U64 cache_key = board->hash ^ ((U64)board->halfmove_clock * 0x9E3779B97F4A7C15ULL);
    uint32_t c_idx = (uint32_t)(cache_key & (EVAL_CACHE_SIZE - 1));
    if (eval_cache[c_idx].hash == cache_key) {
        return eval_cache[c_idx].score;
    }

    uint8_t h_both[2 * SHRA_D] __attribute__((aligned(32)));
    int16_t a1[SHRA_FC1_DIM] __attribute__((aligned(32)));
    int16_t a2[SHRA_FC2_DIM] __attribute__((aligned(32)));

    if (board->turn == WHITE) {
        eval_perspective_avx2(board, WHITE, w_acc, &h_both[0]);
        eval_perspective_avx2(board, BLACK, b_acc, &h_both[SHRA_D]);
    } else {
        eval_perspective_avx2(board, BLACK, b_acc, &h_both[0]);
        eval_perspective_avx2(board, WHITE, w_acc, &h_both[SHRA_D]);
    }

    const __m256i v_ones = _mm256_set1_epi16(1);
    const __m128i v_zero = _mm_setzero_si128();
    const __m128i v_qa   = _mm_set1_epi32(SHRA_QA);

    for (int k = 0; k < SHRA_FC1_DIM; k += 4) {
        const int8_t* w0 = &g_fc1_w[(k + 0) * (2 * SHRA_D)];
        const int8_t* w1 = &g_fc1_w[(k + 1) * (2 * SHRA_D)];
        const int8_t* w2 = &g_fc1_w[(k + 2) * (2 * SHRA_D)];
        const int8_t* w3 = &g_fc1_w[(k + 3) * (2 * SHRA_D)];

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        for (int j = 0; j < 2 * SHRA_D; j += 32) {
            __m256i vh = _mm256_load_si256((const __m256i*)&h_both[j]);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(vh, _mm256_load_si256((const __m256i*)&w0[j])), v_ones));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(vh, _mm256_load_si256((const __m256i*)&w1[j])), v_ones));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(vh, _mm256_load_si256((const __m256i*)&w2[j])), v_ones));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(vh, _mm256_load_si256((const __m256i*)&w3[j])), v_ones));
        }

        __m256i s01 = _mm256_hadd_epi32(acc0, acc1);
        __m256i s23 = _mm256_hadd_epi32(acc2, acc3);
        __m256i s0123 = _mm256_hadd_epi32(s01, s23);
        __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(s0123), _mm256_extracti128_si256(s0123, 1));
        sum4 = _mm_srai_epi32(_mm_add_epi32(sum4, _mm_load_si128((const __m128i*)&g_fc1_b[k])), 6);

        __m128i c = _mm_min_epi32(_mm_max_epi32(sum4, v_zero), v_qa);
        __m128i packed = _mm_packs_epi32(c, c);
        _mm_storel_epi64((__m128i*)&a1[k], packed);
    }

    int total_pieces = count_bits(board->occupied[2]);
    int bucket = (total_pieces - 2) / 4;
    if (bucket < 0) bucket = 0;
    if (bucket >= SHRA_PHASE_BUCKETS) bucket = SHRA_PHASE_BUCKETS - 1;

    const int16_t* fc2_w_b = &g_fc2_w[bucket * SHRA_FC2_DIM * SHRA_FC1_DIM];
    const int32_t* fc2_b_b = &g_fc2_b[bucket * SHRA_FC2_DIM];

    const __m256i va1_0 = _mm256_load_si256((const __m256i*)&a1[0]);
    const __m256i va1_1 = _mm256_load_si256((const __m256i*)&a1[16]);

    for (int k = 0; k < SHRA_FC2_DIM; k += 4) {
        #define FC2_DOT(row, out_v) do { \
            const int16_t* rw = &fc2_w_b[(row) * SHRA_FC1_DIM]; \
            __m256i t0 = _mm256_madd_epi16(va1_0, _mm256_load_si256((const __m256i*)&rw[0])); \
            __m256i t1 = _mm256_madd_epi16(va1_1, _mm256_load_si256((const __m256i*)&rw[16])); \
            (out_v) = _mm256_add_epi32(t0, t1); \
        } while (0)

        __m256i d0, d1, d2, d3;
        FC2_DOT(k + 0, d0);
        FC2_DOT(k + 1, d1);
        FC2_DOT(k + 2, d2);
        FC2_DOT(k + 3, d3);
        #undef FC2_DOT

        __m256i s01 = _mm256_hadd_epi32(d0, d1);
        __m256i s23 = _mm256_hadd_epi32(d2, d3);
        __m256i s0123 = _mm256_hadd_epi32(s01, s23);
        __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(s0123), _mm256_extracti128_si256(s0123, 1));
        sum4 = _mm_srai_epi32(_mm_add_epi32(sum4, _mm_load_si128((const __m128i*)&fc2_b_b[k])), 6);

        __m128i c = _mm_min_epi32(_mm_max_epi32(sum4, v_zero), v_qa);
        __m128i packed = _mm_packs_epi32(c, c);
        _mm_storel_epi64((__m128i*)&a2[k], packed);
    }

    const int16_t* fc3_w_b = &g_fc3_w[bucket * SHRA_FC2_DIM];
    __m256i fc3_0 = _mm256_madd_epi16(_mm256_load_si256((const __m256i*)&a2[0]),
                                      _mm256_load_si256((const __m256i*)&fc3_w_b[0]));
    __m256i fc3_1 = _mm256_madd_epi16(_mm256_load_si256((const __m256i*)&a2[16]),
                                      _mm256_load_si256((const __m256i*)&fc3_w_b[16]));
    int32_t net_out = hsum_epi32(_mm256_add_epi32(fc3_0, fc3_1)) + g_fc3_b[bucket];

    int32_t psqt_out = (board->turn == WHITE)
                     ? ((w_psqt[bucket] - b_psqt[bucket]) / 2)
                     : ((b_psqt[bucket] - w_psqt[bucket]) / 2);

    int64_t total_q = (int64_t)psqt_out + (int64_t)net_out;
    int cp = (int)((total_q * (int64_t)SHRA_EVAL_SCALE) / 8192LL);

    eval_cache[c_idx].hash = cache_key;
    eval_cache[c_idx].score = cp;
    return cp;
}