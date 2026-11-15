#include "defs.h"

U64 pawn_attacks[2][64];
U64 knight_attacks[64];
U64 king_attacks[64];
U64 passed_pawn_masks[2][64];
U64 file_masks[8];
U64 rank_masks[8];
U64 adjacent_files_masks[8];
U64 line_bb[64][64];

static const int knight_offsets[] = { -17, -15, -10, -6, 6, 10, 15, 17 };
static const int king_offsets[]   = { -9, -8, -7, -1, 1, 7, 8, 9 };

U64 bishop_masks[64];
U64 rook_masks[64];
int bishop_shifts[64];
int rook_shifts[64];
U64 bishop_magics[64];
U64 rook_magics[64];

static U64 bishop_table[5248];
static U64 rook_table[102400];
U64* bishop_attacks_ptr[64];
U64* rook_attacks_ptr[64];

static U64 magic_prng_state = 1804289383ULL;
static U64 rand_u64_magic(void) {
    U64 z = (magic_prng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static U64 random_u64_fewbits(void) {
    return rand_u64_magic() & rand_u64_magic() & rand_u64_magic();
}

static U64 mask_bishop_attacks_init(int sq) {
    U64 attacks = 0;
    int r = sq / 8, f = sq % 8;
    for (int dr = 1, df = 1; r + dr < 7 && f + df < 7; ++dr, ++df) attacks |= (1ULL << ((r + dr) * 8 + (f + df)));
    for (int dr = 1, df = -1; r + dr < 7 && f + df > 0; ++dr, --df) attacks |= (1ULL << ((r + dr) * 8 + (f + df)));
    for (int dr = -1, df = 1; r + dr > 0 && f + df < 7; --dr, ++df) attacks |= (1ULL << ((r + dr) * 8 + (f + df)));
    for (int dr = -1, df = -1; r + dr > 0 && f + df > 0; --dr, --df) attacks |= (1ULL << ((r + dr) * 8 + (f + df)));
    return attacks;
}

static U64 mask_rook_attacks_init(int sq) {
    U64 attacks = 0;
    int r = sq / 8, f = sq % 8;
    for (int dr = 1; r + dr < 7; ++dr) attacks |= (1ULL << ((r + dr) * 8 + f));
    for (int dr = -1; r + dr > 0; --dr) attacks |= (1ULL << ((r + dr) * 8 + f));
    for (int df = 1; f + df < 7; ++df) attacks |= (1ULL << (r * 8 + (f + df)));
    for (int df = -1; f + df > 0; --df) attacks |= (1ULL << (r * 8 + (f + df)));
    return attacks;
}

static U64 attacks_on_the_fly_bishop(int sq, U64 block) {
    U64 attacks = 0;
    int r = sq / 8, f = sq % 8;
    for (int dr = 1, df = 1; r + dr < 8 && f + df < 8; ++dr, ++df) { int s = (r + dr) * 8 + (f + df); attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    for (int dr = 1, df = -1; r + dr < 8 && f + df >= 0; ++dr, --df) { int s = (r + dr) * 8 + (f + df); attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    for (int dr = -1, df = 1; r + dr >= 0 && f + df < 8; --dr, ++df) { int s = (r + dr) * 8 + (f + df); attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    for (int dr = -1, df = -1; r + dr >= 0 && f + df >= 0; --dr, --df) { int s = (r + dr) * 8 + (f + df); attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    return attacks;
}

static U64 attacks_on_the_fly_rook(int sq, U64 block) {
    U64 attacks = 0;
    int r = sq / 8, f = sq % 8;
    for (int dr = 1; r + dr < 8; ++dr) { int s = (r + dr) * 8 + f; attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    for (int dr = -1; r + dr >= 0; --dr) { int s = (r + dr) * 8 + f; attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    for (int df = 1; f + df < 8; ++df) { int s = r * 8 + (f + df); attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    for (int df = -1; f + df >= 0; --df) { int s = r * 8 + (f + df); attacks |= (1ULL << s); if (block & (1ULL << s)) break; }
    return attacks;
}

static void init_magic_bitboards(void) {
    U64 occupancies[4096];
    U64 reference_attacks[4096];
    U64 attacks_used[4096];
    bool used[4096];

    int bishop_offset = 0;
    for (int sq = 0; sq < 64; ++sq) {
        bishop_masks[sq] = mask_bishop_attacks_init(sq);
        int bits = count_bits(bishop_masks[sq]);
        bishop_shifts[sq] = 64 - bits;
        int num_subsets = 1 << bits;

        bishop_attacks_ptr[sq] = &bishop_table[bishop_offset];
        bishop_offset += num_subsets;

        U64 occ = 0;
        for (int i = 0; i < num_subsets; ++i) {
            occupancies[i] = occ;
            reference_attacks[i] = attacks_on_the_fly_bishop(sq, occ);
            occ = (occ - bishop_masks[sq]) & bishop_masks[sq];
        }

        while (true) {
            U64 magic = random_u64_fewbits();
            if (count_bits((bishop_masks[sq] * magic) & 0xFF00000000000000ULL) < 6) continue;
            memset(used, 0, num_subsets * sizeof(bool));
            bool fail = false;

            for (int i = 0; i < num_subsets; ++i) {
                int idx = (int)(((occupancies[i] & bishop_masks[sq]) * magic) >> bishop_shifts[sq]);
                if (!used[idx]) {
                    used[idx] = true;
                    attacks_used[idx] = reference_attacks[i];
                } else if (attacks_used[idx] != reference_attacks[i]) {
                    fail = true;
                    break;
                }
            }

            if (!fail) {
                bishop_magics[sq] = magic;
                for (int i = 0; i < num_subsets; ++i) {
                    int idx = (int)(((occupancies[i] & bishop_masks[sq]) * magic) >> bishop_shifts[sq]);
                    bishop_attacks_ptr[sq][idx] = reference_attacks[i];
                }
                break;
            }
        }
    }

    int rook_offset = 0;
    for (int sq = 0; sq < 64; ++sq) {
        rook_masks[sq] = mask_rook_attacks_init(sq);
        int bits = count_bits(rook_masks[sq]);
        rook_shifts[sq] = 64 - bits;
        int num_subsets = 1 << bits;

        rook_attacks_ptr[sq] = &rook_table[rook_offset];
        rook_offset += num_subsets;

        U64 occ = 0;
        for (int i = 0; i < num_subsets; ++i) {
            occupancies[i] = occ;
            reference_attacks[i] = attacks_on_the_fly_rook(sq, occ);
            occ = (occ - rook_masks[sq]) & rook_masks[sq];
        }

        while (true) {
            U64 magic = random_u64_fewbits();
            if (count_bits((rook_masks[sq] * magic) & 0xFF00000000000000ULL) < 6) continue;
            memset(used, 0, num_subsets * sizeof(bool));
            bool fail = false;

            for (int i = 0; i < num_subsets; ++i) {
                int idx = (int)(((occupancies[i] & rook_masks[sq]) * magic) >> rook_shifts[sq]);
                if (!used[idx]) {
                    used[idx] = true;
                    attacks_used[idx] = reference_attacks[i];
                } else if (attacks_used[idx] != reference_attacks[i]) {
                    fail = true;
                    break;
                }
            }

            if (!fail) {
                rook_magics[sq] = magic;
                for (int i = 0; i < num_subsets; ++i) {
                    int idx = (int)(((occupancies[i] & rook_masks[sq]) * magic) >> rook_shifts[sq]);
                    rook_attacks_ptr[sq][idx] = reference_attacks[i];
                }
                break;
            }
        }
    }
}

void init_bitboards(void) {
    for (int f = 0; f < 8; ++f) {
        file_masks[f] = 0x0101010101010101ULL << f;
        adjacent_files_masks[f] = 0;
        if (f > 0) adjacent_files_masks[f] |= file_masks[f - 1];
        if (f < 7) adjacent_files_masks[f] |= file_masks[f + 1];
    }

    for (int r = 0; r < 8; ++r) {
        rank_masks[r] = 0xFFULL << (r * 8);
    }

    for (int sq = 0; sq < 64; ++sq) {
        pawn_attacks[WHITE][sq] = 0;
        pawn_attacks[BLACK][sq] = 0;
        if (sq + 7 < 64 && abs((sq % 8) - ((sq + 7) % 8)) == 1) pawn_attacks[WHITE][sq] |= (1ULL << (sq + 7));
        if (sq + 9 < 64 && abs((sq % 8) - ((sq + 9) % 8)) == 1) pawn_attacks[WHITE][sq] |= (1ULL << (sq + 9));
        if (sq - 7 >= 0 && abs((sq % 8) - ((sq - 7) % 8)) == 1) pawn_attacks[BLACK][sq] |= (1ULL << (sq - 7));
        if (sq - 9 >= 0 && abs((sq % 8) - ((sq - 9) % 8)) == 1) pawn_attacks[BLACK][sq] |= (1ULL << (sq - 9));

        int r = sq / 8, f = sq % 8;
        passed_pawn_masks[WHITE][sq] = 0;
        passed_pawn_masks[BLACK][sq] = 0;

        for (int rank = r + 1; rank < 8; ++rank) {
            passed_pawn_masks[WHITE][sq] |= (1ULL << (rank * 8 + f));
            if (f > 0) passed_pawn_masks[WHITE][sq] |= (1ULL << (rank * 8 + f - 1));
            if (f < 7) passed_pawn_masks[WHITE][sq] |= (1ULL << (rank * 8 + f + 1));
        }

        for (int rank = r - 1; rank >= 0; --rank) {
            passed_pawn_masks[BLACK][sq] |= (1ULL << (rank * 8 + f));
            if (f > 0) passed_pawn_masks[BLACK][sq] |= (1ULL << (rank * 8 + f - 1));
            if (f < 7) passed_pawn_masks[BLACK][sq] |= (1ULL << (rank * 8 + f + 1));
        }
    }

    for (int sq = 0; sq < 64; ++sq) {
        knight_attacks[sq] = 0;
        for (int i = 0; i < 8; ++i) {
            int to = sq + knight_offsets[i];
            if (to >= 0 && to < 64 && abs((sq % 8) - (to % 8)) <= 2) {
                knight_attacks[sq] |= (1ULL << to);
            }
        }
    }

    for (int sq = 0; sq < 64; ++sq) {
        king_attacks[sq] = 0;
        for (int i = 0; i < 8; ++i) {
            int to = sq + king_offsets[i];
            if (to >= 0 && to < 64 && abs((sq % 8) - (to % 8)) <= 1) {
                king_attacks[sq] |= (1ULL << to);
            }
        }
    }

    init_magic_bitboards();

    memset(line_bb, 0, sizeof(line_bb));
    for (int s1 = 0; s1 < 64; ++s1) {
        for (int s2 = 0; s2 < 64; ++s2) {
            if (s1 == s2) continue;
            if (get_rook_attacks(s1, 0ULL) & (1ULL << s2)) {
                line_bb[s1][s2] = (get_rook_attacks(s1, 0ULL) & get_rook_attacks(s2, 0ULL))
                                | (1ULL << s1) | (1ULL << s2);
            } else if (get_bishop_attacks(s1, 0ULL) & (1ULL << s2)) {
                line_bb[s1][s2] = (get_bishop_attacks(s1, 0ULL) & get_bishop_attacks(s2, 0ULL))
                                | (1ULL << s1) | (1ULL << s2);
            }
        }
    }
}

bool is_square_attacked(const Board* board, int sq, int attacker_color) {
    if (sq < 0 || sq >= 64 || (attacker_color != WHITE && attacker_color != BLACK)) return false;
    U64 occ = board->occupied[2];
    int p_off = (attacker_color == WHITE) ? 0 : 6;

    if (pawn_attacks[1 - attacker_color][sq] & board->pieces[p_off + 0]) return true;
    if (knight_attacks[sq] & board->pieces[p_off + 1]) return true;
    if (king_attacks[sq] & board->pieces[p_off + 5]) return true;

    U64 diag = board->pieces[p_off + 2] | board->pieces[p_off + 4];
    if (get_bishop_attacks(sq, occ) & diag) return true;

    U64 straight = board->pieces[p_off + 3] | board->pieces[p_off + 4];
    if (get_rook_attacks(sq, occ) & straight) return true;

    return false;
}

static inline void add_move(MoveList* list, int from, int to, int captured, int promotion, bool ep, bool castle) {
    if (list->count < 256) {
        list->moves[list->count++] = make_move_enc(from, to, promotion, captured, ep, castle);
    }
}

void generate_moves(const Board* board, MoveList* list) {
    list->count = 0;
    int us = board->turn;
    int them = 1 - us;
    int offset = (us == WHITE) ? 0 : 6;
    U64 our_pieces = board->occupied[us];
    U64 their_pieces = board->occupied[them];
    U64 all_occ = board->occupied[2];
    U64 ep_bb = (board->en_passant_square != NO_SQ) ? (1ULL << board->en_passant_square) : 0ULL;

    U64 pawns = board->pieces[offset + PAWN - 1];
    int push_dir = (us == WHITE) ? 8 : -8;
    int promo_rank = (us == WHITE) ? 6 : 1;
    int start_rank = (us == WHITE) ? 1 : 6;

    while (pawns) {
        int from = pop_lsb(&pawns);
        int to = from + push_dir;
        int from_rank = from / 8;
        if (to >= 0 && to < 64 && !(all_occ & (1ULL << to))) {
            if (from_rank == promo_rank) {
                add_move(list, from, to, EMPTY, QUEEN, false, false);
                add_move(list, from, to, EMPTY, ROOK, false, false);
                add_move(list, from, to, EMPTY, BISHOP, false, false);
                add_move(list, from, to, EMPTY, KNIGHT, false, false);
            } else {
                add_move(list, from, to, EMPTY, EMPTY, false, false);
                if (from_rank == start_rank) {
                    int to2 = to + push_dir;
                    if (!(all_occ & (1ULL << to2))) add_move(list, from, to2, EMPTY, EMPTY, false, false);
                }
            }
        }
        U64 attacks = pawn_attacks[us][from] & (their_pieces | ep_bb);
        while (attacks) {
            int cap_to = pop_lsb(&attacks);
            bool is_ep = (cap_to == board->en_passant_square);
            int cap_piece = is_ep ? PAWN : board->board[cap_to];
            if (from_rank == promo_rank) {
                add_move(list, from, cap_to, cap_piece, QUEEN, is_ep, false);
                add_move(list, from, cap_to, cap_piece, ROOK, is_ep, false);
                add_move(list, from, cap_to, cap_piece, BISHOP, is_ep, false);
                add_move(list, from, cap_to, cap_piece, KNIGHT, is_ep, false);
            } else {
                add_move(list, from, cap_to, cap_piece, EMPTY, is_ep, false);
            }
        }
    }

    U64 knights = board->pieces[offset + KNIGHT - 1];
    while (knights) {
        int from = pop_lsb(&knights);
        U64 moves = knight_attacks[from] & ~our_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    U64 bishops = board->pieces[offset + BISHOP - 1];
    while (bishops) {
        int from = pop_lsb(&bishops);
        U64 moves = get_bishop_attacks(from, all_occ) & ~our_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    U64 rooks = board->pieces[offset + ROOK - 1];
    while (rooks) {
        int from = pop_lsb(&rooks);
        U64 moves = get_rook_attacks(from, all_occ) & ~our_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    U64 queens = board->pieces[offset + QUEEN - 1];
    while (queens) {
        int from = pop_lsb(&queens);
        U64 moves = get_queen_attacks(from, all_occ) & ~our_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    int k_from = board->king_sq[us];
    U64 k_moves = king_attacks[k_from] & ~our_pieces;
    while (k_moves) { int to = pop_lsb(&k_moves); add_move(list, k_from, to, board->board[to], EMPTY, false, false); }

    if (us == WHITE) {
        if ((board->castling_rights & WK_CASTLE) && !(all_occ & ((1ULL << F1) | (1ULL << G1))) &&
            !is_square_attacked(board, E1, BLACK) && !is_square_attacked(board, F1, BLACK) && !is_square_attacked(board, G1, BLACK)) {
            add_move(list, E1, G1, EMPTY, EMPTY, false, true);
        }
        if ((board->castling_rights & WQ_CASTLE) && !(all_occ & ((1ULL << D1) | (1ULL << C1) | (1ULL << B1))) &&
            !is_square_attacked(board, E1, BLACK) && !is_square_attacked(board, D1, BLACK) && !is_square_attacked(board, C1, BLACK)) {
            add_move(list, E1, C1, EMPTY, EMPTY, false, true);
        }
    } else {
        if ((board->castling_rights & BK_CASTLE) && !(all_occ & ((1ULL << F8) | (1ULL << G8))) &&
            !is_square_attacked(board, E8, WHITE) && !is_square_attacked(board, F8, WHITE) && !is_square_attacked(board, G8, WHITE)) {
            add_move(list, E8, G8, EMPTY, EMPTY, false, true);
        }
        if ((board->castling_rights & BQ_CASTLE) && !(all_occ & ((1ULL << D8) | (1ULL << C8) | (1ULL << B8))) &&
            !is_square_attacked(board, E8, WHITE) && !is_square_attacked(board, D8, WHITE) && !is_square_attacked(board, C8, WHITE)) {
            add_move(list, E8, C8, EMPTY, EMPTY, false, true);
        }
    }
}

void generate_tactical_moves(const Board* board, MoveList* list) {
    list->count = 0;
    int us = board->turn;
    int them = 1 - us;
    int offset = (us == WHITE) ? 0 : 6;
    U64 their_pieces = board->occupied[them];
    U64 all_occ = board->occupied[2];
    U64 ep_bb = (board->en_passant_square != NO_SQ) ? (1ULL << board->en_passant_square) : 0ULL;
    U64 enemy_pawns = board->pieces[(us == WHITE ? 6 : 0) + PAWN - 1];

    U64 pawns = board->pieces[offset + PAWN - 1];
    int push_dir = (us == WHITE) ? 8 : -8;
    int promo_rank = (us == WHITE) ? 6 : 1;
    int near_promo_rank = (us == WHITE) ? 5 : 2;

    while (pawns) {
        int from = pop_lsb(&pawns);
        int from_rank = from / 8;

        if (from_rank == promo_rank) {
            int to = from + push_dir;
            if (to >= 0 && to < 64 && !(all_occ & (1ULL << to))) {
                add_move(list, from, to, EMPTY, QUEEN, false, false);
                add_move(list, from, to, EMPTY, ROOK, false, false);
                add_move(list, from, to, EMPTY, BISHOP, false, false);
                add_move(list, from, to, EMPTY, KNIGHT, false, false);
            }
        } else if (from_rank == near_promo_rank) {
            int to = from + push_dir;
            if (!(all_occ & (1ULL << to)) && (enemy_pawns & passed_pawn_masks[us][to]) == 0ULL) {
                add_move(list, from, to, EMPTY, EMPTY, false, false);
            }
        }

        U64 attacks = pawn_attacks[us][from] & (their_pieces | ep_bb);
        while (attacks) {
            int cap_to = pop_lsb(&attacks);
            bool is_ep = (cap_to == board->en_passant_square);
            int cap_piece = is_ep ? PAWN : board->board[cap_to];
            if (from_rank == promo_rank) {
                add_move(list, from, cap_to, cap_piece, QUEEN, is_ep, false);
                add_move(list, from, cap_to, cap_piece, ROOK, is_ep, false);
                add_move(list, from, cap_to, cap_piece, BISHOP, is_ep, false);
                add_move(list, from, cap_to, cap_piece, KNIGHT, is_ep, false);
            } else {
                add_move(list, from, cap_to, cap_piece, EMPTY, is_ep, false);
            }
        }
    }

    U64 knights = board->pieces[offset + KNIGHT - 1];
    while (knights) {
        int from = pop_lsb(&knights);
        U64 moves = knight_attacks[from] & their_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    U64 bishops = board->pieces[offset + BISHOP - 1];
    while (bishops) {
        int from = pop_lsb(&bishops);
        U64 moves = get_bishop_attacks(from, all_occ) & their_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    U64 rooks = board->pieces[offset + ROOK - 1];
    while (rooks) {
        int from = pop_lsb(&rooks);
        U64 moves = get_rook_attacks(from, all_occ) & their_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    U64 queens = board->pieces[offset + QUEEN - 1];
    while (queens) {
        int from = pop_lsb(&queens);
        U64 moves = get_queen_attacks(from, all_occ) & their_pieces;
        while (moves) { int to = pop_lsb(&moves); add_move(list, from, to, board->board[to], EMPTY, false, false); }
    }

    int k_from = board->king_sq[us];
    U64 k_moves = king_attacks[k_from] & their_pieces;
    while (k_moves) { int to = pop_lsb(&k_moves); add_move(list, k_from, to, board->board[to], EMPTY, false, false); }
}

static int get_smallest_attacker(const Board* board, int sq, int attacker_color, U64* attacked_by) {
    if (attacker_color != WHITE && attacker_color != BLACK) return NO_SQ;
    int p_off = (attacker_color == WHITE) ? 0 : 6;
    U64 occ = board->occupied[2] & ~(*attacked_by);

    U64 p_atk = pawn_attacks[1 - attacker_color][sq] & board->pieces[p_off + 0] & ~(*attacked_by);
    if (p_atk) return pop_lsb(&p_atk);

    U64 n_atk = knight_attacks[sq] & board->pieces[p_off + 1] & ~(*attacked_by);
    if (n_atk) return pop_lsb(&n_atk);

    U64 b_atk = get_bishop_attacks(sq, occ) & board->pieces[p_off + 2] & ~(*attacked_by);
    if (b_atk) return pop_lsb(&b_atk);

    U64 r_atk = get_rook_attacks(sq, occ) & board->pieces[p_off + 3] & ~(*attacked_by);
    if (r_atk) return pop_lsb(&r_atk);

    U64 q_atk = get_queen_attacks(sq, occ) & board->pieces[p_off + 4] & ~(*attacked_by);
    if (q_atk) return pop_lsb(&q_atk);

    U64 k_atk = king_attacks[sq] & board->pieces[p_off + 5] & ~(*attacked_by);
    if (k_atk) return pop_lsb(&k_atk);

    return NO_SQ;
}

int see(const Board* board, Move move) {
    if (move == MOVE_NONE) return 0;
    int from = move_from(move);
    int to = move_to(move);
    int cap = move_captured(move);
    int promo = move_promo(move);

    int attacker_color = board->colors[from];
    if (attacker_color != WHITE && attacker_color != BLACK) return 0;

    int gain[32];
    int d = 0;
    U64 attacked_by = 0;

    gain[d] = (cap != EMPTY) ? piece_values[cap] : 0;
    if (promo != EMPTY) {
        gain[d] += piece_values[promo] - piece_values[PAWN];
    }

    int current_piece = (promo != EMPTY) ? promo : board->board[from];
    int current_color = 1 - attacker_color;

    attacked_by |= (1ULL << from);
    if (move_is_ep(move)) {
        int ep_sq = to + ((attacker_color == WHITE) ? -8 : 8);
        attacked_by |= (1ULL << ep_sq);
    }

    while (d < 31) {
        d++;
        int attacker_sq = get_smallest_attacker(board, to, current_color, &attacked_by);
        if (attacker_sq == NO_SQ) {
            d--;
            break;
        }

        if (board->board[attacker_sq] == KING) {
            if (get_smallest_attacker(board, to, 1 - current_color, &attacked_by) != NO_SQ) {
                d--;
                break;
            }
        }

        gain[d] = piece_values[current_piece] - gain[d - 1];
        current_piece = board->board[attacker_sq];
        attacked_by |= (1ULL << attacker_sq);
        current_color = 1 - current_color;
    }

    while (d > 0) {
        if (-gain[d] < gain[d - 1]) gain[d - 1] = -gain[d];
        d--;
    }

    return gain[0];
}