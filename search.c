#include "defs.h"
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CORRHIST_LIMIT (256 * 256)

int g_num_threads = 1;
static int lmr_table[MAX_PLY][64];
static const int lmp_move_count[6] = { 0, 5, 8, 13, 19, 26 };

typedef struct {
    Board board;
    Move killer_moves[MAX_PLY][2];
    int history_moves[2][64][64];
    int capture_history[7][64][7];
    int cont_history[2][7][64][7][64];
    Move counter_moves[64][64];
    int eval_stack[MAX_PLY];

    int pawn_corrhist[2][CORRHIST_SIZE];
    int non_pawn_corrhist[2][2][CORRHIST_SIZE];

    Move pv_table[MAX_PLY][MAX_PLY];
    int pv_length[MAX_PLY];

    int16_t w_acc_stack[MAX_PLY][SHRA_D] __attribute__((aligned(32)));
    int16_t b_acc_stack[MAX_PLY][SHRA_D] __attribute__((aligned(32)));
    int32_t w_psqt_stack[MAX_PLY][SHRA_PHASE_BUCKETS] __attribute__((aligned(32)));
    int32_t b_psqt_stack[MAX_PLY][SHRA_PHASE_BUCKETS] __attribute__((aligned(32)));
    bool acc_valid[MAX_PLY];
    Move acc_move[MAX_PLY];
    int acc_moving_piece[MAX_PLY];
    int acc_moving_color[MAX_PLY];
    int acc_king_sq[MAX_PLY][2];

    ShraEvalCacheEntry eval_cache[EVAL_CACHE_SIZE];

    uint64_t nodes;
    int seldepth;
    int root_depth;
    int thread_id;
} ThreadData;

static ThreadData* g_threads[MAX_THREADS] = {NULL};

static inline int clamp_int(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static ThreadData* get_thread_data(int tid) {
    if (!g_threads[tid]) {
        g_threads[tid] = (ThreadData*)_mm_malloc(sizeof(ThreadData), 32);
        if (!g_threads[tid]) {
            fprintf(stderr, "Failed to allocate search thread memory\n");
            exit(EXIT_FAILURE);
        }
        memset(g_threads[tid], 0, sizeof(ThreadData));
    }
    return g_threads[tid];
}

void init_search_tables(void) {
    for (int d = 0; d < MAX_PLY; ++d) {
        for (int m = 0; m < 64; ++m) {
            if (d >= 2 && m >= 2) {
                lmr_table[d][m] = (int)(0.60 + log((double)d) * log((double)m) / 2.10);
            } else {
                lmr_table[d][m] = 0;
            }
        }
    }
}

void clear_search_history(void) {
    for (int t = 0; t < MAX_THREADS; ++t) {
        if (g_threads[t]) {
            memset(g_threads[t]->killer_moves, 0, sizeof(g_threads[t]->killer_moves));
            memset(g_threads[t]->history_moves, 0, sizeof(g_threads[t]->history_moves));
            memset(g_threads[t]->capture_history, 0, sizeof(g_threads[t]->capture_history));
            memset(g_threads[t]->cont_history, 0, sizeof(g_threads[t]->cont_history));
            memset(g_threads[t]->counter_moves, 0, sizeof(g_threads[t]->counter_moves));
            memset(g_threads[t]->pawn_corrhist, 0, sizeof(g_threads[t]->pawn_corrhist));
            memset(g_threads[t]->non_pawn_corrhist, 0, sizeof(g_threads[t]->non_pawn_corrhist));
            memset(g_threads[t]->eval_cache, 0, sizeof(g_threads[t]->eval_cache));
        }
    }
}

static inline uint64_t get_total_nodes(int nthreads) {
    if (nthreads <= 1) {
        return __atomic_load_n(&g_threads[0]->nodes, __ATOMIC_RELAXED);
    }
    uint64_t total = 0;
    for (int t = 0; t < nthreads; ++t) {
        if (g_threads[t]) {
            total += __atomic_load_n(&g_threads[t]->nodes, __ATOMIC_RELAXED);
        }
    }
    return total;
}

static inline void check_time_and_input(SearchInfo* info, int nthreads) {
    if (info->max_nodes > 0 && get_total_nodes(nthreads) >= info->max_nodes) {
        __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
        return;
    }
    if (get_time_ms() >= info->stop_time) {
        __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
        return;
    }
    if (input_waiting()) {
        char input[256];
        if (fgets(input, sizeof(input), stdin)) {
            if (strncmp(input, "isready", 7) == 0) {
                printf("readyok\n");
                fflush(stdout);
            } else if (strncmp(input, "stop", 4) == 0) {
                __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
            } else if (strncmp(input, "quit", 4) == 0) {
                __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
                exit(0);
            }
        }
    }
}

static inline void ensure_accumulator(ThreadData* td, int ply) {
    if (ply <= 0 || td->acc_valid[ply]) return;

    int valid_p = ply - 1;
    while (valid_p > 0 && !td->acc_valid[valid_p]) {
        valid_p--;
    }
    if (valid_p < 0) valid_p = 0;

    for (int k = valid_p + 1; k <= ply; ++k) {
        if (k <= 0 || k >= MAX_PLY) break;
        if (td->acc_move[k] == MOVE_NONE) {
            memcpy(td->w_acc_stack[k], td->w_acc_stack[k - 1], SHRA_D * sizeof(int16_t));
            memcpy(td->b_acc_stack[k], td->b_acc_stack[k - 1], SHRA_D * sizeof(int16_t));
            memcpy(td->w_psqt_stack[k], td->w_psqt_stack[k - 1], SHRA_PHASE_BUCKETS * sizeof(int32_t));
            memcpy(td->b_psqt_stack[k], td->b_psqt_stack[k - 1], SHRA_PHASE_BUCKETS * sizeof(int32_t));
        } else {
            shra_update_accumulators_move(td->acc_move[k],
                                          td->acc_moving_piece[k],
                                          td->acc_moving_color[k],
                                          td->acc_king_sq[k][WHITE],
                                          td->acc_king_sq[k][BLACK],
                                          td->w_acc_stack[k - 1], td->b_acc_stack[k - 1],
                                          td->w_psqt_stack[k - 1], td->b_psqt_stack[k - 1],
                                          td->w_acc_stack[k], td->b_acc_stack[k],
                                          td->w_psqt_stack[k], td->b_psqt_stack[k]);
        }
        td->acc_valid[k] = true;
    }
}

static inline void record_move_acc(ThreadData* td, int ply, Move move, int moving_piece, int moving_color) {
    const Board* board = &td->board;
    int new_w_ksq = board->king_sq[WHITE];
    int new_b_ksq = board->king_sq[BLACK];
    td->acc_king_sq[ply + 1][WHITE] = new_w_ksq;
    td->acc_king_sq[ply + 1][BLACK] = new_b_ksq;

    bool refresh_needed = (moving_piece == KING) &&
        (shra_king_bucket_id(move_from(move), moving_color) !=
         shra_king_bucket_id(move_to(move), moving_color));

    if (refresh_needed) {
        ensure_accumulator(td, ply);
        if (moving_color == WHITE) {
            shra_init_single_accumulator(board, WHITE, td->w_acc_stack[ply + 1], td->w_psqt_stack[ply + 1]);
            shra_update_single_move(move, moving_piece, moving_color, BLACK, new_b_ksq,
                                    td->b_acc_stack[ply], td->b_psqt_stack[ply],
                                    td->b_acc_stack[ply + 1], td->b_psqt_stack[ply + 1]);
        } else {
            shra_init_single_accumulator(board, BLACK, td->b_acc_stack[ply + 1], td->b_psqt_stack[ply + 1]);
            shra_update_single_move(move, moving_piece, moving_color, WHITE, new_w_ksq,
                                    td->w_acc_stack[ply], td->w_psqt_stack[ply],
                                    td->w_acc_stack[ply + 1], td->w_psqt_stack[ply + 1]);
        }
        td->acc_valid[ply + 1] = true;
    } else {
        td->acc_valid[ply + 1] = false;
        td->acc_move[ply + 1] = move;
        td->acc_moving_piece[ply + 1] = moving_piece;
        td->acc_moving_color[ply + 1] = moving_color;
    }
}

static inline int evaluate_raw_node(ThreadData* td, int ply) {
    const Board* board = &td->board;
    U64 cache_key = board->hash ^ ((U64)board->halfmove_clock * 0x9E3779B97F4A7C15ULL);
    uint32_t c_idx = (uint32_t)(cache_key & (EVAL_CACHE_SIZE - 1));
    if (td->eval_cache[c_idx].hash == cache_key) {
        return td->eval_cache[c_idx].score;
    }

    ensure_accumulator(td, ply);
    return shra_evaluate_board(board,
                               td->w_acc_stack[ply], td->b_acc_stack[ply],
                               td->w_psqt_stack[ply], td->b_psqt_stack[ply],
                               td->eval_cache);
}

static inline int adjust_eval_with_corrhist(const ThreadData* td, int raw_eval) {
    const Board* b = &td->board;
    int us = b->turn;
    int p_corr = td->pawn_corrhist[us][b->pawn_hash & (CORRHIST_SIZE - 1)];
    int w_corr = td->non_pawn_corrhist[us][WHITE][b->non_pawn_hash[WHITE] & (CORRHIST_SIZE - 1)];
    int b_corr = td->non_pawn_corrhist[us][BLACK][b->non_pawn_hash[BLACK] & (CORRHIST_SIZE - 1)];
    int corr = (p_corr + w_corr + b_corr) / 1024;
    return clamp_int(raw_eval + corr, -MATE_SCORE + MAX_PLY + 1, MATE_SCORE - MAX_PLY - 1);
}

static inline void update_corrhist(ThreadData* td, int depth, int diff) {
    const Board* b = &td->board;
    int us = b->turn;
    int w = (depth > 16) ? 16 : depth;
    if (w < 1) w = 1;
    int scaled_diff = clamp_int(diff, -256, 256) * 256;

    int* p_entry = &td->pawn_corrhist[us][b->pawn_hash & (CORRHIST_SIZE - 1)];
    *p_entry = clamp_int((*p_entry * (128 - w) + scaled_diff * w) / 128, -CORRHIST_LIMIT, CORRHIST_LIMIT);

    int* w_entry = &td->non_pawn_corrhist[us][WHITE][b->non_pawn_hash[WHITE] & (CORRHIST_SIZE - 1)];
    *w_entry = clamp_int((*w_entry * (128 - w) + scaled_diff * w) / 128, -CORRHIST_LIMIT, CORRHIST_LIMIT);

    int* b_entry = &td->non_pawn_corrhist[us][BLACK][b->non_pawn_hash[BLACK] & (CORRHIST_SIZE - 1)];
    *b_entry = clamp_int((*b_entry * (128 - w) + scaled_diff * w) / 128, -CORRHIST_LIMIT, CORRHIST_LIMIT);
}

static inline int evaluate_node(ThreadData* td, int ply) {
    int raw = evaluate_raw_node(td, ply);
    return adjust_eval_with_corrhist(td, raw);
}

static inline bool is_insufficient_material(const Board* board) {
    if (board->pieces[W_PAWN] | board->pieces[B_PAWN] |
        board->pieces[W_ROOK] | board->pieces[B_ROOK] |
        board->pieces[W_QUEEN] | board->pieces[B_QUEEN]) {
        return false;
    }

    int total = count_bits(board->occupied[2]);
    if (total <= 3) return true;

    if (total == 4 && count_bits(board->pieces[W_BISHOP]) == 1 && count_bits(board->pieces[B_BISHOP]) == 1) {
        const U64 light_squares = 0x55AA55AA55AA55AAULL;
        bool w_light = (board->pieces[W_BISHOP] & light_squares) != 0;
        bool b_light = (board->pieces[B_BISHOP] & light_squares) != 0;
        if (w_light == b_light) return true;
    }
    return false;
}

static inline bool move_gives_check_fast(const Board* board, Move move) {
    int us = board->turn;
    int them = 1 - us;
    int enemy_k_sq = board->king_sq[them];
    if (enemy_k_sq < 0 || enemy_k_sq >= 64) return false;

    int promo = move_promo(move);
    int piece = (promo != EMPTY) ? promo : board->board[move_from(move)];
    int to = move_to(move);

    switch (piece) {
        case KNIGHT:
            return (knight_attacks[to] & (1ULL << enemy_k_sq)) != 0;
        case BISHOP:
            return (get_bishop_attacks(to, board->occupied[2]) & (1ULL << enemy_k_sq)) != 0;
        case ROOK:
            return (get_rook_attacks(to, board->occupied[2]) & (1ULL << enemy_k_sq)) != 0;
        case QUEEN:
            return (get_queen_attacks(to, board->occupied[2]) & (1ULL << enemy_k_sq)) != 0;
        case PAWN:
            return (pawn_attacks[us][to] & (1ULL << enemy_k_sq)) != 0;
        default:
            return false;
    }
}

static inline bool is_passed_pawn_push(const Board* board, Move move) {
    int from = move_from(move);
    if (board->board[from] != PAWN) return false;

    int us = board->colors[from];
    int to = move_to(move);
    int p_idx = (us == WHITE ? 6 : 0) + PAWN - 1;
    return (board->pieces[p_idx] & passed_pawn_masks[us][to]) == 0ULL;
}

static bool is_move_strictly_legal(Board* board, Move m) {
    if (m == MOVE_NONE) return false;
    int from = move_from(m);
    int to = move_to(m);
    if (from == to) return false;

    MoveList list;
    generate_moves(board, &list);
    for (int i = 0; i < list.count; ++i) {
        if (same_move_ft_p(list.moves[i], m)) {
            if (make_move(board, list.moves[i], 0, 0)) {
                unmake_move(board, list.moves[i], 0);
                return true;
            }
            return false;
        }
    }
    return false;
}

static void score_moves(ThreadData* td, MoveList* list, int ply, Move tt_move,
                        Move prev_m1, int prev_piece, Move prev_m2, int prev2_piece) {
    const Board* board = &td->board;
    int us = board->turn;
    int prev_to = (prev_m1 != MOVE_NONE) ? move_to(prev_m1) : 0;
    int prev2_to = (prev_m2 != MOVE_NONE) ? move_to(prev_m2) : 0;
    Move counter_m = (prev_m1 != MOVE_NONE) ? td->counter_moves[move_from(prev_m1)][prev_to] : MOVE_NONE;
    Move k0 = td->killer_moves[ply][0];
    Move k1 = td->killer_moves[ply][1];

    for (int i = 0; i < list->count; ++i) {
        Move move = list->moves[i];
        int from = move_from(move);
        int to = move_to(move);
        int promo = move_promo(move);
        int cap = move_captured(move);
        int piece = board->board[from];

        if (tt_move != MOVE_NONE && same_move_ft_p(move, tt_move)) {
            list->scores[i] = 60000;
        } else if (cap != EMPTY) {
            int cap_hist = 0;
            if (piece >= PAWN && piece <= KING && cap >= PAWN && cap <= KING) {
                cap_hist = td->capture_history[piece][to][cap];
            }
            int see_score = (piece_values[cap] < piece_values[piece]) ? see(board, move) : 0;
            if (see_score >= 0) {
                int s = 30000 + (piece_values[cap] * 10) - piece_values[piece] + see_score + (cap_hist / 16);
                if (promo == QUEEN) s += 20000;
                list->scores[i] = s;
            } else {
                list->scores[i] = -10000 + see_score + (cap_hist / 16);
            }
        } else if (promo != EMPTY) {
            if (promo == QUEEN) list->scores[i] = 28000;
            else if (promo == KNIGHT) list->scores[i] = 19000;
            else list->scores[i] = -5000;
        } else {
            bool scored = false;
            if (move_gives_check_fast(board, move)) {
                if (!is_square_attacked(board, to, 1 - us) || see(board, move) >= 0) {
                    list->scores[i] = 26000;
                    scored = true;
                }
            }
            if (!scored) {
                if (move_is_castle(move)) {
                    list->scores[i] = 25000;
                } else if (is_passed_pawn_push(board, move)) {
                    list->scores[i] = 22000;
                } else if (k0 != MOVE_NONE && same_move_ft_p(move, k0)) {
                    list->scores[i] = 20000;
                } else if (k1 != MOVE_NONE && same_move_ft_p(move, k1)) {
                    list->scores[i] = 18000;
                } else if (counter_m != MOVE_NONE && same_move_ft_p(move, counter_m)) {
                    list->scores[i] = 16000;
                } else {
                    int h = td->history_moves[us][from][to];
                    if (piece >= PAWN && piece <= KING) {
                        if (prev_piece >= PAWN && prev_piece <= KING) {
                            h += td->cont_history[0][prev_piece][prev_to][piece][to];
                        }
                        if (prev2_piece >= PAWN && prev2_piece <= KING) {
                            h += td->cont_history[1][prev2_piece][prev2_to][piece][to] / 2;
                        }
                    }
                    list->scores[i] = h;
                }
            }
        }
    }
}

static void score_qmoves(const ThreadData* td, MoveList* list, Move tt_move, int* see_scores) {
    const Board* board = &td->board;
    for (int i = 0; i < list->count; ++i) {
        Move move = list->moves[i];
        int from = move_from(move);
        int to = move_to(move);
        int cap = move_captured(move);
        int promo = move_promo(move);
        int piece = board->board[from];

        int s_val = (promo != EMPTY || (cap != EMPTY && piece_values[cap] < piece_values[piece]))
                    ? see(board, move)
                    : (piece_values[cap] - piece_values[piece]);
        see_scores[i] = s_val;

        if (tt_move != MOVE_NONE && same_move_ft_p(move, tt_move)) {
            list->scores[i] = 60000;
        } else if (s_val >= 0) {
            int cap_hist = (cap != EMPTY && piece >= PAWN && piece <= KING && cap >= PAWN && cap <= KING)
                           ? td->capture_history[piece][to][cap] : 0;
            int s = 30000 + (piece_values[cap] * 10) - piece_values[piece] + s_val + (cap_hist / 16);
            if (promo == QUEEN) s += 20000;
            list->scores[i] = s;
        } else {
            list->scores[i] = -10000 + s_val;
        }
    }
}

static inline Move pick_next_move(MoveList* list, int current_idx, int* out_score) {
    int best_idx = current_idx;
    int best_score = list->scores[current_idx];

    for (int i = current_idx + 1; i < list->count; ++i) {
        if (list->scores[i] > best_score) {
            best_score = list->scores[i];
            best_idx = i;
        }
    }

    if (best_idx != current_idx) {
        Move tmp_m = list->moves[current_idx];
        list->moves[current_idx] = list->moves[best_idx];
        list->moves[best_idx] = tmp_m;

        int tmp_s = list->scores[current_idx];
        list->scores[current_idx] = list->scores[best_idx];
        list->scores[best_idx] = tmp_s;
    }

    *out_score = best_score;
    return list->moves[current_idx];
}

static inline Move pick_next_move_q(MoveList* list, int* see_scores, int current_idx,
                                    int* out_score, int* out_see) {
    int best_idx = current_idx;
    int best_score = list->scores[current_idx];

    for (int i = current_idx + 1; i < list->count; ++i) {
        if (list->scores[i] > best_score) {
            best_score = list->scores[i];
            best_idx = i;
        }
    }

    if (best_idx != current_idx) {
        Move tmp_m = list->moves[current_idx];
        list->moves[current_idx] = list->moves[best_idx];
        list->moves[best_idx] = tmp_m;

        int tmp_s = list->scores[current_idx];
        list->scores[current_idx] = list->scores[best_idx];
        list->scores[best_idx] = tmp_s;

        int tmp_see = see_scores[current_idx];
        see_scores[current_idx] = see_scores[best_idx];
        see_scores[best_idx] = tmp_see;
    }

    *out_score = best_score;
    *out_see = see_scores[current_idx];
    return list->moves[current_idx];
}

static int quiescence(ThreadData* td, SearchInfo* info, int alpha, int beta,
                      int ply, int qply, bool is_pv, int nthreads) {
    if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;

    if (info->max_nodes > 0 && nthreads == 1 && td->nodes >= info->max_nodes) {
        __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
        return 0;
    }

    if (ply >= MAX_PLY - 1) return evaluate_node(td, ply);
    if (ply > td->seldepth) td->seldepth = ply;

    uint64_t new_nodes = td->nodes + 1;
    __atomic_store_n(&td->nodes, new_nodes, __ATOMIC_RELAXED);

    if ((new_nodes & 2047) == 0) {
        if (td->thread_id == 0) check_time_and_input(info, nthreads);
        if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;
    } else if (info->max_nodes > 0 && nthreads > 1 && (new_nodes & 255) == 0 && td->thread_id == 0) {
        if (get_total_nodes(nthreads) >= info->max_nodes) {
            __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
            return 0;
        }
    }

    Board* board = &td->board;
    if (is_insufficient_material(board)) return 0;

    Move tt_move = MOVE_NONE;
    int tt_s = 0, tt_d = 0, tt_f = 0;
    bool tt_cut = false;
    probe_tt_full(board->hash, 0, alpha, beta, ply, &tt_s, &tt_move, &tt_d, &tt_f, &tt_cut);
    (void)tt_d;
    (void)tt_f;
    if (tt_cut && !is_pv) {
        return tt_s;
    }

    bool in_check = is_square_attacked(board, board->king_sq[board->turn], 1 - board->turn);
    if (in_check && qply >= 16) {
        return evaluate_node(td, ply);
    }

    int orig_alpha = alpha;
    int best_score = -INF;
    int stand_pat = -INF;

    if (!in_check) {
        stand_pat = evaluate_node(td, ply);
        if (stand_pat >= beta) {
            store_tt(board->hash, 0, TT_BETA, stand_pat, MOVE_NONE, ply);
            return stand_pat;
        }
        best_score = stand_pat;
        if (alpha < stand_pat) alpha = stand_pat;
    }

    MoveList list;
    int see_scores[256];
    if (in_check) {
        generate_moves(board, &list);
        score_moves(td, &list, ply, tt_move, MOVE_NONE, EMPTY, MOVE_NONE, EMPTY);
    } else {
        generate_tactical_moves(board, &list);
        score_qmoves(td, &list, tt_move, see_scores);
    }

    int legal_moves = 0;
    Move best_move = MOVE_NONE;

    for (int i = 0; i < list.count; ++i) {
        int move_score = 0;
        int see_val = 0;
        Move current_move = in_check
                            ? pick_next_move(&list, i, &move_score)
                            : pick_next_move_q(&list, see_scores, i, &move_score, &see_val);
        (void)move_score;

        int from = move_from(current_move);
        int cap = move_captured(current_move);
        int promo = move_promo(current_move);

        if (!in_check) {
            if (see_val < 0) continue;
            int gain_est = piece_values[cap] + ((promo != EMPTY) ? (piece_values[promo] - piece_values[PAWN]) : 0);
            if (stand_pat + gain_est + 200 < alpha) continue;
        } else {
            if (qply >= 6 && legal_moves > 0 && cap == EMPTY && promo == EMPTY && see(board, current_move) < 0) {
                continue;
            }
        }

        int moving_piece = board->board[from];
        int moving_color = board->colors[from];

        if (!make_move(board, current_move, ply, td->thread_id)) continue;
        legal_moves++;

        record_move_acc(td, ply, current_move, moving_piece, moving_color);

        int score = -quiescence(td, info, -beta, -alpha, ply + 1, qply + 1, is_pv, nthreads);
        unmake_move(board, current_move, ply);

        if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;

        if (score > best_score) {
            best_score = score;
            if (score > alpha) {
                alpha = score;
                best_move = current_move;
                if (score >= beta) {
                    store_tt(board->hash, 0, TT_BETA, best_score, best_move, ply);
                    return best_score;
                }
            }
        }
    }

    if (in_check && legal_moves == 0) return -MATE_SCORE + ply;

    int tt_flag = (best_score > orig_alpha) ? TT_EXACT : TT_ALPHA;
    store_tt(board->hash, 0, tt_flag, best_score, best_move, ply);
    return best_score;
}

static int negamax(ThreadData* td, SearchInfo* info, int depth, int ply, int alpha, int beta,
                   bool is_pv, Move prev_m1, int prev_piece, Move prev_m2, int prev2_piece,
                   Move excluded_move, bool allow_nmp, int nthreads) {
    if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;

    if (info->max_nodes > 0 && nthreads == 1 && td->nodes >= info->max_nodes) {
        __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
        return 0;
    }

    if (ply >= MAX_PLY - 1) return evaluate_node(td, ply);

    if (excluded_move == MOVE_NONE) {
        td->pv_length[ply] = ply;
    }
    if (ply > td->seldepth) td->seldepth = ply;

    if (depth <= 0) return quiescence(td, info, alpha, beta, ply, 0, is_pv, nthreads);

    uint64_t new_nodes = td->nodes + 1;
    __atomic_store_n(&td->nodes, new_nodes, __ATOMIC_RELAXED);

    if ((new_nodes & 2047) == 0) {
        if (td->thread_id == 0) check_time_and_input(info, nthreads);
        if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;
    } else if (info->max_nodes > 0 && nthreads > 1 && (new_nodes & 255) == 0 && td->thread_id == 0) {
        if (get_total_nodes(nthreads) >= info->max_nodes) {
            __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
            return 0;
        }
    }

    Board* board = &td->board;

    if (ply > 0 && excluded_move == MOVE_NONE) {
        if (board->halfmove_clock >= 100) return 0;
        if (is_insufficient_material(board)) return 0;

        for (int i = ply - 2; i >= 0; i -= 2) {
            if (board->history_hash[i] == board->hash) return 0;
        }

        int limit = board->game_ply - board->halfmove_clock;
        if (limit < 0) limit = 0;
        for (int i = board->game_ply - 1; i >= limit; --i) {
            if (board->game_history[i] == board->hash) return 0;
        }
    }

    bool tt_hit = false;
    Move tt_move = MOVE_NONE;
    int tt_score = 0;
    int tt_depth = -1;
    int tt_flags = 0;
    bool tt_cutoff = false;

    if (excluded_move == MOVE_NONE) {
        tt_hit = probe_tt_full(board->hash, depth, alpha, beta, ply,
                               &tt_score, &tt_move, &tt_depth, &tt_flags, &tt_cutoff);
        if (tt_cutoff && ply > 0 && !is_pv) {
            return tt_score;
        }
    }

    bool in_check = is_square_attacked(board, board->king_sq[board->turn], 1 - board->turn);
    if (in_check && ply < td->root_depth * 2 && depth < MAX_PLY - 4) {
        depth++;
    }

    if (!is_pv && !in_check && excluded_move == MOVE_NONE && depth >= 4 && tt_move == MOVE_NONE) {
        depth--;
    }

    int static_eval = 0;
    int eval = 0;
    bool improving = false;
    if (!in_check) {
        static_eval = evaluate_node(td, ply);
        if (excluded_move == MOVE_NONE) {
            td->eval_stack[ply] = static_eval;
        }
        improving = (ply >= 2 && td->eval_stack[ply - 2] != -INF && static_eval > td->eval_stack[ply - 2]);

        eval = static_eval;
        if (tt_hit && abs(tt_score) < MATE_SCORE - MAX_PLY) {
            if ((tt_flags == TT_EXACT) ||
                (tt_flags == TT_BETA && tt_score > eval) ||
                (tt_flags == TT_ALPHA && tt_score < eval)) {
                eval = tt_score;
            }
        }
    } else if (excluded_move == MOVE_NONE) {
        td->eval_stack[ply] = -INF;
    }

    if (!is_pv && !in_check && excluded_move == MOVE_NONE && depth <= 2 && eval + 350 * depth <= alpha) {
        int r_score = quiescence(td, info, alpha, alpha + 1, ply, 0, false, nthreads);
        if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;
        if (r_score <= alpha) return r_score;
    }

    if (!is_pv && !in_check && excluded_move == MOVE_NONE && depth <= 6) {
        int rfp_margin = (improving ? 80 : 110) * depth;
        if (eval - rfp_margin >= beta) return eval - rfp_margin;
    }

    if (!is_pv && !in_check && allow_nmp && excluded_move == MOVE_NONE &&
        ply > 0 && depth >= 3 && prev_m1 != MOVE_NONE && eval >= beta) {
        int us_off = (board->turn == WHITE) ? 0 : 6;
        U64 non_pawns = board->pieces[us_off + 1] | board->pieces[us_off + 2] |
                        board->pieces[us_off + 3] | board->pieces[us_off + 4];
        if (non_pawns != 0ULL) {
            int orig_ep = board->en_passant_square;
            int orig_turn = board->turn;

            board->history_hash[ply] = board->hash;
            board->turn = 1 - board->turn;
            board->hash ^= side_key;
            if (board->en_passant_square != NO_SQ) {
                board->hash ^= ep_keys[board->en_passant_square];
                board->en_passant_square = NO_SQ;
            }

            td->acc_valid[ply + 1] = false;
            td->acc_move[ply + 1] = MOVE_NONE;
            td->acc_king_sq[ply + 1][WHITE] = board->king_sq[WHITE];
            td->acc_king_sq[ply + 1][BLACK] = board->king_sq[BLACK];

            int r = 3 + depth / 4 + (improving ? 1 : 0);
            int score = -negamax(td, info, depth - 1 - r, ply + 1, -beta, -beta + 1,
                                 false, MOVE_NONE, EMPTY, prev_m1, prev_piece,
                                 MOVE_NONE, true, nthreads);

            board->turn = orig_turn;
            board->hash ^= side_key;
            board->en_passant_square = orig_ep;
            if (board->en_passant_square != NO_SQ) board->hash ^= ep_keys[board->en_passant_square];

            if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;

            if (score >= beta) {
                if (score >= MATE_SCORE - MAX_PLY) score = beta;
                int np_count = count_bits(non_pawns);
                if (depth < 12 && np_count > 1) {
                    return score;
                }
                int v_score = negamax(td, info, depth - 1 - r, ply, beta - 1, beta,
                                      false, prev_m1, prev_piece, prev_m2, prev2_piece,
                                      MOVE_NONE, false, nthreads);
                if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;
                if (v_score >= beta) return score;
            }
        }
    }

    if (!is_pv && !in_check && excluded_move == MOVE_NONE && depth >= 5 && abs(beta) < MATE_SCORE - MAX_PLY) {
        int pc_beta = beta + 180;
        int pc_depth = depth - 4;
        int req_see = pc_beta - eval;
        if (req_see < 1) req_see = 1;

        MoveList pc_list;
        int pc_see[256];
        generate_tactical_moves(board, &pc_list);
        score_qmoves(td, &pc_list, tt_move, pc_see);

        for (int i = 0; i < pc_list.count; ++i) {
            int pc_ms = 0;
            int pc_s_val = 0;
            Move pc_m = pick_next_move_q(&pc_list, pc_see, i, &pc_ms, &pc_s_val);
            (void)pc_ms;
            if (move_captured(pc_m) == EMPTY && move_promo(pc_m) == EMPTY) continue;
            if (pc_s_val < req_see) continue;

            int from = move_from(pc_m);
            int mp = board->board[from];
            int mc = board->colors[from];
            if (!make_move(board, pc_m, ply, td->thread_id)) continue;

            record_move_acc(td, ply, pc_m, mp, mc);

            int pc_score = -negamax(td, info, pc_depth, ply + 1, -pc_beta, -pc_beta + 1,
                                    false, pc_m, mp, prev_m1, prev_piece,
                                    MOVE_NONE, true, nthreads);
            unmake_move(board, pc_m, ply);

            if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;
            if (pc_score >= pc_beta) {
                store_tt(board->hash, pc_depth, TT_BETA, pc_score, pc_m, ply);
                return pc_score;
            }
        }
    }

    MoveList list;
    generate_moves(board, &list);
    score_moves(td, &list, ply, tt_move, prev_m1, prev_piece, prev_m2, prev2_piece);

    int legal_moves = 0;
    int best_score = -INF;
    Move best_move_node = MOVE_NONE;
    int tt_flag = TT_ALPHA;
    Move quiet_moves_tried[64];
    int quiet_pieces_tried[64];
    int quiet_count = 0;
    int prev_to = (prev_m1 != MOVE_NONE) ? move_to(prev_m1) : 0;
    int prev2_to = (prev_m2 != MOVE_NONE) ? move_to(prev_m2) : 0;

    bool can_singular = (excluded_move == MOVE_NONE && ply > 0 && ply < MAX_PLY - 4 &&
                         depth >= 7 && tt_hit && tt_move != MOVE_NONE &&
                         tt_depth >= depth - 3 && tt_flags != TT_ALPHA &&
                         abs(tt_score) < MATE_SCORE - MAX_PLY);

    for (int i = 0; i < list.count; ++i) {
        int move_score = 0;
        Move current_move = pick_next_move(&list, i, &move_score);

        if (excluded_move != MOVE_NONE && same_move_ft_p(current_move, excluded_move)) {
            continue;
        }

        int from = move_from(current_move);
        int to = move_to(current_move);
        int promo = move_promo(current_move);
        int cap = move_captured(current_move);

        bool is_quiet = (cap == EMPTY && promo == EMPTY);
        bool is_passer = is_quiet && is_passed_pawn_push(board, current_move);
        bool is_castling = move_is_castle(current_move);
        bool gives_check = move_gives_check_fast(board, current_move);
        bool bad_check = (is_quiet && gives_check &&
                          is_square_attacked(board, to, 1 - board->turn) &&
                          see(board, current_move) < 0);
        int to_rank = to / 8;

        bool is_advanced_passer = is_passer && ((board->turn == WHITE && to_rank == 6) ||
                                                (board->turn == BLACK && to_rank == 1));

        if (!is_pv && !in_check && legal_moves > 0) {
            if (is_quiet && (!gives_check || bad_check) && !is_passer && !is_castling) {
                if (depth <= 5 && legal_moves >= lmp_move_count[depth]) continue;
                if (depth <= 4 && eval + 120 * depth <= alpha) continue;
            }

            if (is_quiet && !is_advanced_passer && depth <= 5 && (legal_moves >= 2 || bad_check)) {
                if (board->board[from] > PAWN &&
                    is_square_attacked(board, to, 1 - board->turn) &&
                    see(board, current_move) < -80 * depth - 60) {
                    continue;
                }
            }

            if (cap != EMPTY && promo == EMPTY && !gives_check && depth <= 5) {
                if (piece_values[cap] < piece_values[board->board[from]] &&
                    see(board, current_move) < -90 * depth) {
                    continue;
                }
            }
        }

        int extension = 0;
        if (can_singular && same_move_ft_p(current_move, tt_move)) {
            int s_beta = tt_score - 2 * depth;
            int s_depth = (depth - 1) / 2;
            int se_score = negamax(td, info, s_depth, ply, s_beta - 1, s_beta,
                                   false, prev_m1, prev_piece, prev_m2, prev2_piece,
                                   current_move, true, nthreads);
            if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;

            if (se_score < s_beta) {
                extension = 1;
                if (!is_pv && !in_check && ply < td->root_depth * 2 &&
                    se_score < s_beta - 25 && depth < 14) {
                    extension = 2;
                }
            } else if (s_beta >= beta) {
                return s_beta;
            }
        } else if (is_advanced_passer && depth <= 6 && ply < td->root_depth * 2 &&
                   !in_check && see(board, current_move) >= 0) {
            extension = 1;
        }

        int moving_piece = board->board[from];
        int moving_color = board->colors[from];

        if (!make_move(board, current_move, ply, td->thread_id)) continue;
        legal_moves++;

        record_move_acc(td, ply, current_move, moving_piece, moving_color);

        if (is_quiet && quiet_count < 64) {
            quiet_moves_tried[quiet_count] = current_move;
            quiet_pieces_tried[quiet_count] = moving_piece;
            quiet_count++;
        }

        int score;
        int next_depth = depth - 1 + extension;

        if (legal_moves == 1) {
            score = -negamax(td, info, next_depth, ply + 1, -beta, -alpha,
                             is_pv, current_move, moving_piece, prev_m1, prev_piece,
                             MOVE_NONE, true, nthreads);
        } else {
            int reduction = 0;
            if (depth >= 3 && legal_moves >= 3 && is_quiet && !in_check &&
                (!gives_check || bad_check) && !is_advanced_passer && !is_castling) {
                int d_idx = (depth < MAX_PLY) ? depth : MAX_PLY - 1;
                int m_idx = (legal_moves < 64) ? legal_moves : 63;
                reduction = lmr_table[d_idx][m_idx];

                if (!improving) reduction += 1;
                if (!is_pv) reduction += 1;
                if (move_score < 0) reduction += 1;
                if (is_passer && reduction > 1) reduction -= 1;

                if (reduction < 1) reduction = 1;
                if (next_depth <= 1) reduction = 0;
                else if (reduction > next_depth - 1) reduction = next_depth - 1;
            }

            score = -negamax(td, info, next_depth - reduction, ply + 1, -alpha - 1, -alpha,
                             false, current_move, moving_piece, prev_m1, prev_piece,
                             MOVE_NONE, true, nthreads);

            if (!__atomic_load_n(&info->stopped, __ATOMIC_RELAXED) && score > alpha && reduction > 0) {
                score = -negamax(td, info, next_depth, ply + 1, -alpha - 1, -alpha,
                                 false, current_move, moving_piece, prev_m1, prev_piece,
                                 MOVE_NONE, true, nthreads);
            }
            if (!__atomic_load_n(&info->stopped, __ATOMIC_RELAXED) && score > alpha && is_pv) {
                score = -negamax(td, info, next_depth, ply + 1, -beta, -alpha,
                                 true, current_move, moving_piece, prev_m1, prev_piece,
                                 MOVE_NONE, true, nthreads);
            }
        }

        unmake_move(board, current_move, ply);
        if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) return 0;

        if (score > best_score) {
            best_score = score;
            if (score > alpha) {
                alpha = score;
                best_move_node = current_move;
                tt_flag = TT_EXACT;

                if (is_pv && excluded_move == MOVE_NONE && ply < MAX_PLY - 1) {
                    td->pv_table[ply][ply] = current_move;
                    int next_len = td->pv_length[ply + 1];
                    if (next_len > MAX_PLY) next_len = MAX_PLY;
                    for (int next_ply = ply + 1; next_ply < next_len; ++next_ply) {
                        td->pv_table[ply][next_ply] = td->pv_table[ply + 1][next_ply];
                    }
                    td->pv_length[ply] = (next_len > ply + 1) ? next_len : (ply + 1);
                }

                if (score >= beta) {
                    if (is_quiet) {
                        if (td->killer_moves[ply][0] == MOVE_NONE ||
                            !same_move_ft_p(td->killer_moves[ply][0], current_move)) {
                            td->killer_moves[ply][1] = td->killer_moves[ply][0];
                            td->killer_moves[ply][0] = current_move;
                        }

                        int bonus = depth * depth;
                        int* h_ptr = &td->history_moves[board->turn][from][to];
                        *h_ptr += bonus - (*h_ptr * abs(bonus)) / 16384;

                        if (moving_piece >= PAWN && moving_piece <= KING) {
                            if (prev_piece >= PAWN && prev_piece <= KING) {
                                int* ch_ptr = &td->cont_history[0][prev_piece][prev_to][moving_piece][to];
                                *ch_ptr += bonus - (*ch_ptr * abs(bonus)) / 16384;
                            }
                            if (prev2_piece >= PAWN && prev2_piece <= KING) {
                                int* ch2_ptr = &td->cont_history[1][prev2_piece][prev2_to][moving_piece][to];
                                *ch2_ptr += bonus - (*ch2_ptr * abs(bonus)) / 16384;
                            }
                        }

                        for (int q = 0; q < quiet_count - 1; ++q) {
                            Move bad_m = quiet_moves_tried[q];
                            int bf = move_from(bad_m);
                            int bt = move_to(bad_m);
                            int bp = quiet_pieces_tried[q];

                            int* bh = &td->history_moves[board->turn][bf][bt];
                            *bh -= bonus + (*bh * abs(bonus)) / 16384;

                            if (bp >= PAWN && bp <= KING) {
                                if (prev_piece >= PAWN && prev_piece <= KING) {
                                    int* bch = &td->cont_history[0][prev_piece][prev_to][bp][bt];
                                    *bch -= bonus + (*bch * abs(bonus)) / 16384;
                                }
                                if (prev2_piece >= PAWN && prev2_piece <= KING) {
                                    int* bch2 = &td->cont_history[1][prev2_piece][prev2_to][bp][bt];
                                    *bch2 -= bonus + (*bch2 * abs(bonus)) / 16384;
                                }
                            }
                        }

                        if (prev_m1 != MOVE_NONE) {
                            td->counter_moves[move_from(prev_m1)][prev_to] = current_move;
                        }
                    } else if (cap != EMPTY) {
                        int bonus = depth * depth * 2;
                        if (bonus > 2000) bonus = 2000;
                        if (moving_piece >= PAWN && moving_piece <= KING && cap >= PAWN && cap <= KING) {
                            int* ch = &td->capture_history[moving_piece][to][cap];
                            *ch += bonus - (*ch * abs(bonus)) / 16384;
                        }
                    }

                    if (!in_check && excluded_move == MOVE_NONE &&
                        is_quiet && abs(best_score) < MATE_SCORE - MAX_PLY &&
                        best_score > static_eval) {
                        update_corrhist(td, depth, best_score - static_eval);
                    }

                    if (excluded_move == MOVE_NONE) {
                        store_tt(board->hash, depth, TT_BETA, best_score, best_move_node, ply);
                    }
                    return best_score;
                }
            }
        }
    }

    if (legal_moves == 0) {
        if (excluded_move != MOVE_NONE) return alpha;
        return in_check ? (-MATE_SCORE + ply) : 0;
    }

    if (excluded_move == MOVE_NONE) {
        if (!in_check && abs(best_score) < MATE_SCORE - MAX_PLY &&
            (best_move_node == MOVE_NONE ||
             (move_captured(best_move_node) == EMPTY && move_promo(best_move_node) == EMPTY))) {
            if ((tt_flag == TT_EXACT) ||
                (tt_flag == TT_ALPHA && best_score < static_eval)) {
                update_corrhist(td, depth, best_score - static_eval);
            }
        }
        store_tt(board->hash, depth, tt_flag, best_score, best_move_node, ply);
    }
    return best_score;
}

static void prepare_thread_state(ThreadData* td, const Board* board, int tid,
                                 const int16_t* base_w, const int16_t* base_b,
                                 const int32_t* base_pw, const int32_t* base_pb) {
    td->board = *board;
    td->thread_id = tid;
    td->nodes = 0;
    td->seldepth = 0;
    td->root_depth = 1;

    memset(td->killer_moves, 0, sizeof(td->killer_moves));
    memset(td->eval_stack, 0, sizeof(td->eval_stack));
    memset(td->pv_table, 0, sizeof(td->pv_table));
    memset(td->pv_length, 0, sizeof(td->pv_length));

    for (int c = 0; c < 2; ++c) {
        for (int f = 0; f < 64; ++f) {
            for (int t = 0; t < 64; ++t) {
                td->history_moves[c][f][t] /= 2;
            }
        }
    }
    for (int p = 0; p < 7; ++p) {
        for (int s = 0; s < 64; ++s) {
            for (int cp = 0; cp < 7; ++cp) {
                td->capture_history[p][s][cp] /= 2;
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int p1 = 0; p1 < 7; ++p1) {
            for (int s1 = 0; s1 < 64; ++s1) {
                for (int p2 = 0; p2 < 7; ++p2) {
                    for (int s2 = 0; s2 < 64; ++s2) {
                        td->cont_history[i][p1][s1][p2][s2] /= 2;
                    }
                }
            }
        }
    }

    memcpy(td->w_acc_stack[0], base_w, SHRA_D * sizeof(int16_t));
    memcpy(td->b_acc_stack[0], base_b, SHRA_D * sizeof(int16_t));
    memcpy(td->w_psqt_stack[0], base_pw, SHRA_PHASE_BUCKETS * sizeof(int32_t));
    memcpy(td->b_psqt_stack[0], base_pb, SHRA_PHASE_BUCKETS * sizeof(int32_t));
    td->acc_valid[0] = true;
    td->acc_move[0] = MOVE_NONE;
    td->acc_king_sq[0][WHITE] = board->king_sq[WHITE];
    td->acc_king_sq[0][BLACK] = board->king_sq[BLACK];
}

Move search_root(Board* board, SearchInfo* info) {
    int nthreads = g_num_threads;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;

    __atomic_store_n(&info->stopped, false, __ATOMIC_RELEASE);
    info->seldepth = 0;

    ThreadData* td0 = get_thread_data(0);
    shra_init_accumulators(board, td0->w_acc_stack[0], td0->b_acc_stack[0],
                           td0->w_psqt_stack[0], td0->b_psqt_stack[0]);
    prepare_thread_state(td0, board, 0,
                         td0->w_acc_stack[0], td0->b_acc_stack[0],
                         td0->w_psqt_stack[0], td0->b_psqt_stack[0]);

    for (int t = 1; t < nthreads; ++t) {
        ThreadData* tdh = get_thread_data(t);
        prepare_thread_state(tdh, board, t,
                             td0->w_acc_stack[0], td0->b_acc_stack[0],
                             td0->w_psqt_stack[0], td0->b_psqt_stack[0]);
    }

    Move best_move_overall = MOVE_NONE;
    MoveList root_moves;
    generate_moves(board, &root_moves);

    for (int i = 0; i < root_moves.count; ++i) {
        if (make_move(board, root_moves.moves[i], 0, 0)) {
            best_move_overall = root_moves.moves[i];
            unmake_move(board, root_moves.moves[i], 0);
            break;
        }
    }

    if (best_move_overall == MOVE_NONE) return MOVE_NONE;

    #pragma omp parallel num_threads(nthreads) if(nthreads > 1)
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif

        ThreadData* td = g_threads[tid];

        if (tid == 0) {
            int best_score = 0;
            int prev_score = 0;
            Move prev_best_move = MOVE_NONE;
            int stability = 0;

            for (int depth = 1; depth <= info->depth; ++depth) {
                td->root_depth = depth;
                int alpha = -INF;
                int beta  = INF;
                int delta = 25;

                if (depth >= 4 && abs(best_score) < MATE_SCORE - MAX_PLY) {
                    alpha = best_score - delta;
                    beta  = best_score + delta;
                }

                while (true) {
                    int score = negamax(td, info, depth, 0, alpha, beta, true,
                                        MOVE_NONE, EMPTY, MOVE_NONE, EMPTY,
                                        MOVE_NONE, true, nthreads);
                    if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) break;

                    best_score = score;

                    if (best_score <= alpha) {
                        delta += delta;
                        alpha = (delta > 500 || best_score - delta < -INF + 1000) ? -INF : (best_score - delta);
                    } else if (best_score >= beta) {
                        delta += delta;
                        beta = (delta > 500 || best_score + delta > INF - 1000) ? INF : (best_score + delta);
                    } else {
                        break;
                    }
                }

                if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) break;

                if (td->pv_length[0] > 0 && is_move_strictly_legal(&td->board, td->pv_table[0][0])) {
                    best_move_overall = td->pv_table[0][0];
                }

                if (depth > 1 && same_move_ft_p(best_move_overall, prev_best_move)) {
                    stability++;
                } else {
                    stability = 0;
                }

                uint64_t now = get_time_ms();
                uint64_t elapsed_ms = (now > info->start_time) ? (now - info->start_time) : 1;
                uint64_t total_nodes = get_total_nodes(nthreads);
                uint64_t nps = (total_nodes * 1000ULL) / elapsed_ms;
                int hashfull = get_hashfull();

                char pv_str[1024] = {0};
                for (int i = 0; i < td->pv_length[0] && i < MAX_PLY; ++i) {
                    Move m = td->pv_table[0][i];
                    if (m == MOVE_NONE) break;

                    int mf = move_from(m);
                    int mt = move_to(m);
                    int mp = move_promo(m);
                    char move_str[10];
                    char promo = '\0';
                    if (mp == QUEEN) promo = 'q';
                    else if (mp == ROOK) promo = 'r';
                    else if (mp == BISHOP) promo = 'b';
                    else if (mp == KNIGHT) promo = 'n';

                    if (promo) {
                        snprintf(move_str, sizeof(move_str), "%c%c%c%c%c ",
                            'a' + (mf % 8), '1' + (mf / 8),
                            'a' + (mt % 8), '1' + (mt / 8), promo);
                    } else {
                        snprintf(move_str, sizeof(move_str), "%c%c%c%c ",
                            'a' + (mf % 8), '1' + (mf / 8),
                            'a' + (mt % 8), '1' + (mt / 8));
                    }
                    if ((int)strlen(pv_str) + (int)strlen(move_str) < (int)sizeof(pv_str) - 1) {
                        strcat(pv_str, move_str);
                    }
                }

                if (best_score > MATE_SCORE - MAX_PLY) {
                    int mate_in_moves = (MATE_SCORE - best_score + 1) / 2;
                    printf("info depth %d seldepth %d score mate %d nodes %llu nps %llu hashfull %d time %llu pv %s\n",
                           depth, td->seldepth, mate_in_moves,
                           (unsigned long long)total_nodes, (unsigned long long)nps,
                           hashfull, (unsigned long long)elapsed_ms, pv_str);
                } else if (best_score < -MATE_SCORE + MAX_PLY) {
                    int mate_in_moves = (-MATE_SCORE - best_score - 1) / 2;
                    printf("info depth %d seldepth %d score mate %d nodes %llu nps %llu hashfull %d time %llu pv %s\n",
                           depth, td->seldepth, mate_in_moves,
                           (unsigned long long)total_nodes, (unsigned long long)nps,
                           hashfull, (unsigned long long)elapsed_ms, pv_str);
                } else {
                    printf("info depth %d seldepth %d score cp %d nodes %llu nps %llu hashfull %d time %llu pv %s\n",
                           depth, td->seldepth, best_score,
                           (unsigned long long)total_nodes, (unsigned long long)nps,
                           hashfull, (unsigned long long)elapsed_ms, pv_str);
                }
                fflush(stdout);

                if (best_score > MATE_SCORE - MAX_PLY && (MATE_SCORE - best_score) <= depth) break;
                if (info->max_nodes > 0 && total_nodes >= info->max_nodes) break;

                if (depth > 1 && info->stop_time > info->start_time) {
                    double alloc = (double)(info->stop_time - info->start_time);
                    double soft_ratio;
                    if (info->movetime > 0) {
                        soft_ratio = 0.90;
                    } else {
                        soft_ratio = 0.60;
                        if (depth >= 5) {
                            if (stability == 0) soft_ratio *= 1.25;
                            else if (stability >= 4 && abs(prev_score - best_score) < 15) soft_ratio *= 0.75;
                            if (prev_score - best_score > 25) soft_ratio *= 1.30;
                        }
                        if (soft_ratio > 0.85) soft_ratio = 0.85;
                    }
                    if ((double)elapsed_ms >= alloc * soft_ratio) break;
                }

                prev_best_move = best_move_overall;
                prev_score = best_score;
            }

            info->seldepth = td->seldepth;
            __atomic_store_n(&info->stopped, true, __ATOMIC_RELEASE);
        } else {
            int helper_best = 0;
            int start_d = 1 + (tid % 2);
            for (int depth = start_d; depth <= info->depth; ++depth) {
                td->root_depth = depth;
                int alpha = -INF;
                int beta  = INF;
                int delta = 25 + (tid % 3) * 5;

                if (depth >= 4 && abs(helper_best) < MATE_SCORE - MAX_PLY) {
                    alpha = helper_best - delta;
                    beta  = helper_best + delta;
                }

                while (true) {
                    int score = negamax(td, info, depth, 0, alpha, beta, true,
                                        MOVE_NONE, EMPTY, MOVE_NONE, EMPTY,
                                        MOVE_NONE, true, nthreads);
                    if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) break;

                    helper_best = score;

                    if (helper_best <= alpha) {
                        delta += delta;
                        alpha = (delta > 500 || helper_best - delta < -INF + 1000) ? -INF : (helper_best - delta);
                    } else if (helper_best >= beta) {
                        delta += delta;
                        beta = (delta > 500 || helper_best + delta > INF - 1000) ? INF : (helper_best + delta);
                    } else {
                        break;
                    }
                }
                if (__atomic_load_n(&info->stopped, __ATOMIC_RELAXED)) break;
            }
        }
    }

    if (!is_move_strictly_legal(board, best_move_overall)) {
        for (int i = 0; i < root_moves.count; ++i) {
            if (make_move(board, root_moves.moves[i], 0, 0)) {
                best_move_overall = root_moves.moves[i];
                unmake_move(board, root_moves.moves[i], 0);
                break;
            }
        }
    }

    return best_move_overall;
}
