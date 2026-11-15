#ifndef DEFS_H
#define DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define ENGINE_NAME "Phoenix 8"
#define ENGINE_AUTHOR "Matheus Campos"
#define MAX_PLY 128
#define MAX_THREADS 32
#define MATE_SCORE 100000
#define INF 300000
#define DEFAULT_TT_SIZE_MB 128
#define DEFAULT_MOVE_OVERHEAD 30

// SHRA v5 (Split-Stream Quantized INT16/UINT8/INT8)
#define SHRA_KING_BUCKETS 12
#define SHRA_FEAT_PER_BUCKET 768
#define SHRA_NUM_KING_FEATS (SHRA_KING_BUCKETS * SHRA_FEAT_PER_BUCKET)
#define SHRA_NUM_GLOBAL_FEATS 6
#define SHRA_TOTAL_INPUTS (SHRA_NUM_KING_FEATS + SHRA_NUM_GLOBAL_FEATS)
#define SHRA_GLOBAL_D 512
#define SHRA_CHANNELS 8
#define SHRA_SPATIAL_D (64 * SHRA_CHANNELS)
#define SHRA_D (SHRA_GLOBAL_D + SHRA_SPATIAL_D)
#define SHRA_PHASE_BUCKETS 8
#define SHRA_FC1_DIM 32
#define SHRA_FC2_DIM 32
#define SHRA_QA 128
#define SHRA_QW 64
#define SHRA_EVAL_SCALE 410

#define EVAL_CACHE_SIZE (1 << 17)
#define CORRHIST_SIZE (1 << 14)

typedef unsigned long long U64;
#define C64(c) c##ULL

static inline int pop_lsb(U64* bb) {
    int lsb = __builtin_ctzll(*bb);
    *bb &= *bb - 1;
    return lsb;
}

static inline int count_bits(U64 bb) {
    return __builtin_popcountll(bb);
}

enum { 
    A1, B1, C1, D1, E1, F1, G1, H1, 
    A2, B2, C2, D2, E2, F2, G2, H2, 
    A3, B3, C3, D3, E3, F3, G3, H3, 
    A4, B4, C4, D4, E4, F4, G4, H4, 
    A5, B5, C5, D5, E5, F5, G5, H5, 
    A6, B6, C6, D6, E6, F6, G6, H6, 
    A7, B7, C7, D7, E7, F7, G7, H7, 
    A8, B8, C8, D8, E8, F8, G8, H8, NO_SQ 
};

enum { EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { WHITE, BLACK };
enum { WK_CASTLE = 1, WQ_CASTLE = 2, BK_CASTLE = 4, BQ_CASTLE = 8 };
enum { TT_EXACT = 1, TT_ALPHA = 2, TT_BETA = 3 };

enum {
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_COUNT = 14
};

typedef uint32_t Move;
#define MOVE_NONE 0U

static inline Move make_move_enc(int from, int to, int promo, int cap, bool ep, bool castle) {
    return (uint32_t)from
         | ((uint32_t)to << 6)
         | ((uint32_t)promo << 12)
         | ((uint32_t)cap << 15)
         | ((uint32_t)(ep ? 1 : 0) << 18)
         | ((uint32_t)(castle ? 1 : 0) << 19);
}

static inline int  move_from(Move m)      { return (int)(m & 0x3F); }
static inline int  move_to(Move m)        { return (int)((m >> 6) & 0x3F); }
static inline int  move_promo(Move m)     { return (int)((m >> 12) & 0x7); }
static inline int  move_captured(Move m)  { return (int)((m >> 15) & 0x7); }
static inline bool move_is_ep(Move m)     { return ((m >> 18) & 1) != 0; }
static inline bool move_is_castle(Move m) { return ((m >> 19) & 1) != 0; }
static inline bool same_move_ft_p(Move a, Move b) {
    return ((a ^ b) & 0x7FFFU) == 0;
}

typedef struct { 
    Move moves[256]; 
    int scores[256];
    int count; 
} MoveList;

typedef struct {
    U64 pieces[12];
    U64 occupied[3];
    int board[64];
    int colors[64];
    int turn;
    int castling_rights;
    int en_passant_square;
    int king_sq[2];
    U64 hash;
    U64 pawn_hash;
    U64 non_pawn_hash[2];
    int halfmove_clock;
    int history_castling[MAX_PLY];
    int history_en_passant[MAX_PLY];
    int history_halfmove[MAX_PLY];
    U64 history_hash[MAX_PLY];
    U64 history_pawn_hash[MAX_PLY];
    U64 history_non_pawn_hash[MAX_PLY][2];
    U64 game_history[2048];
    int game_ply;
    int ply_count;
} Board;

typedef struct {
    uint64_t start_time;
    uint64_t stop_time;
    volatile bool stopped;
    uint64_t max_nodes;
    int depth;
    int seldepth;
    int wtime;
    int btime;
    int winc;
    int binc;
    int movestogo;
    int movetime;
} SearchInfo;

typedef struct {
    U64 hash;
    int score;
} ShraEvalCacheEntry;

extern int g_num_threads;
extern int g_move_overhead;
extern const int piece_values[PIECE_COUNT];
extern U64 piece_keys[14][64];
extern U64 side_key;
extern U64 castle_keys[16];
extern U64 ep_keys[64];

extern U64 pawn_attacks[2][64];
extern U64 knight_attacks[64];
extern U64 king_attacks[64];
extern U64 passed_pawn_masks[2][64];
extern U64 file_masks[8];
extern U64 rank_masks[8];
extern U64 adjacent_files_masks[8];
extern U64 line_bb[64][64];

extern U64 bishop_masks[64];
extern U64 rook_masks[64];
extern int bishop_shifts[64];
extern int rook_shifts[64];
extern U64 bishop_magics[64];
extern U64 rook_magics[64];
extern U64* bishop_attacks_ptr[64];
extern U64* rook_attacks_ptr[64];

static inline U64 get_bishop_attacks(int sq, U64 occ) {
    occ &= bishop_masks[sq];
    occ *= bishop_magics[sq];
    occ >>= bishop_shifts[sq];
    return bishop_attacks_ptr[sq][occ];
}

static inline U64 get_rook_attacks(int sq, U64 occ) {
    occ &= rook_masks[sq];
    occ *= rook_magics[sq];
    occ >>= rook_shifts[sq];
    return rook_attacks_ptr[sq][occ];
}

static inline U64 get_queen_attacks(int sq, U64 occ) {
    return get_bishop_attacks(sq, occ) | get_rook_attacks(sq, occ);
}

uint64_t get_time_ms(void);
int input_waiting(void);

void init_zobrist(void);
void init_bitboards(void);
void init_search_tables(void);
void clear_search_history(void);
U64 generate_hash(const Board* board);
void uci_loop(void);
void reset_board(Board* board);
void parse_fen(Board* board, const char* fen);
bool make_move(Board* board, Move move, int ply, int thread_id);
void unmake_move(Board* board, Move move, int ply);
void generate_moves(const Board* board, MoveList* list);
void generate_tactical_moves(const Board* board, MoveList* list);
bool is_square_attacked(const Board* board, int sq, int attacker_color);
Move search_root(Board* board, SearchInfo* info);

int see(const Board* board, Move move);

bool init_shra_embedded(void);
bool init_shra(const char* bin_path);
int shra_king_bucket_id(int ksq, int color);
void shra_init_single_accumulator(const Board* board, int side, int16_t* out_acc, int32_t* out_psqt);
void shra_init_accumulators(const Board* board, int16_t* out_w, int16_t* out_b, int32_t* out_psqt_w, int32_t* out_psqt_b);
void shra_update_single_move(Move move, int piece, int color, int side, int side_ksq,
                             const int16_t* in_acc, const int32_t* in_psqt,
                             int16_t* out_acc, int32_t* out_psqt);
void shra_update_accumulators_move(Move move, int piece, int color, int w_ksq, int b_ksq,
                                   const int16_t* in_w, const int16_t* in_b,
                                   const int32_t* in_psqt_w, const int32_t* in_psqt_b,
                                   int16_t* out_w, int16_t* out_b,
                                   int32_t* out_psqt_w, int32_t* out_psqt_b);
int shra_evaluate_board(const Board* board,
                        const int16_t* w_acc, const int16_t* b_acc,
                        const int32_t* w_psqt, const int32_t* b_psqt,
                        ShraEvalCacheEntry* eval_cache);

void init_tt(int size_mb);
void clear_tt(void);
void new_search_tt(void);
bool probe_tt_full(U64 hash, int depth, int alpha, int beta, int ply,
                   int* out_score, Move* out_tt_move,
                   int* out_tt_depth, int* out_tt_flags, bool* out_cutoff);
void store_tt(U64 hash, int depth, int flags, int score, Move best, int ply);
int get_hashfull(void);

#endif