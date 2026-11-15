#include "defs.h"

U64 piece_keys[14][64];
U64 side_key;
U64 castle_keys[16];
U64 ep_keys[64];

static U64 splitmix64_state = 123456789ULL;

static U64 rand_u64(void) {
    U64 z = (splitmix64_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void init_zobrist(void) {
    splitmix64_state = 123456789ULL;
    for (int p = PAWN; p <= KING; ++p) {
        for (int sq = 0; sq < 64; ++sq) {
            piece_keys[p * 2][sq] = rand_u64();
            piece_keys[p * 2 + 1][sq] = rand_u64();
        }
    }
    side_key = rand_u64();
    for (int i = 0; i < 16; ++i) castle_keys[i] = rand_u64();
    for (int i = 0; i < 64; ++i) ep_keys[i] = rand_u64();
}

U64 generate_hash(const Board* board) {
    U64 hash = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (board->board[sq] != EMPTY) {
            hash ^= piece_keys[board->board[sq] * 2 + board->colors[sq]][sq];
        }
    }
    if (board->turn == BLACK) hash ^= side_key;
    if (board->en_passant_square != NO_SQ) hash ^= ep_keys[board->en_passant_square];
    hash ^= castle_keys[board->castling_rights];
    return hash;
}