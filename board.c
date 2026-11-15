#include "defs.h"

void reset_board(Board* board) {
    parse_fen(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void parse_fen(Board* board, const char* fen) {
    memset(board, 0, sizeof(Board));
    for (int i = 0; i < 64; ++i) { board->colors[i] = -1; }
    board->en_passant_square = NO_SQ;
    board->halfmove_clock = 0;
    board->ply_count = 0;

    int rank = 7, file = 0;
    const char* ptr = fen;

    while (*ptr != ' ' && *ptr != '\0') {
        char c = *ptr++;
        if (c >= '1' && c <= '8') {
            file += c - '0';
        } else if (c == '/') {
            rank--;
            file = 0;
        } else if (rank >= 0 && rank < 8 && file >= 0 && file < 8) {
            int sq = rank * 8 + file;
            int color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
            char upper_c = (c >= 'a') ? (char)(c - 32) : c;
            int piece = EMPTY;
            switch (upper_c) {
                case 'P': piece = PAWN; break;
                case 'N': piece = KNIGHT; break;
                case 'B': piece = BISHOP; break;
                case 'R': piece = ROOK; break;
                case 'Q': piece = QUEEN; break;
                case 'K': piece = KING; board->king_sq[color] = sq; break;
            }
            if (piece != EMPTY) {
                board->board[sq] = piece;
                board->colors[sq] = color;

                int b_idx = (color == WHITE ? 0 : 6) + piece - 1;
                board->pieces[b_idx] |= (1ULL << sq);
                board->occupied[color] |= (1ULL << sq);
            }
            file++;
        }
    }
    board->occupied[2] = board->occupied[0] | board->occupied[1];

    if (*ptr == ' ') ptr++;
    if (*ptr != '\0') {
        board->turn = (*ptr++ == 'w') ? WHITE : BLACK;
    }

    if (*ptr == ' ') ptr++;
    while (*ptr != ' ' && *ptr != '\0') {
        char c = *ptr++;
        if (c == 'K') board->castling_rights |= WK_CASTLE;
        if (c == 'Q') board->castling_rights |= WQ_CASTLE;
        if (c == 'k') board->castling_rights |= BK_CASTLE;
        if (c == 'q') board->castling_rights |= BQ_CASTLE;
    }

    if (*ptr == ' ') ptr++;
    if (*ptr != '\0') {
        if (*ptr != '-') {
            if (ptr[0] >= 'a' && ptr[0] <= 'h' && ptr[1] >= '1' && ptr[1] <= '8') {
                board->en_passant_square = (ptr[0] - 'a') + (ptr[1] - '1') * 8;
                ptr += 2;
            } else {
                while (*ptr != ' ' && *ptr != '\0') ptr++;
            }
        } else {
            ptr++;
        }
        if (*ptr == ' ') ptr++;
    }

    int halfmove = 0, fullmove = 1;
    if (*ptr != '\0' && sscanf(ptr, "%d %d", &halfmove, &fullmove) >= 1) {
        board->halfmove_clock = halfmove;
        board->ply_count = (fullmove - 1) * 2;
        if (board->turn == BLACK) board->ply_count++;
    }

    board->hash = generate_hash(board);
    board->pawn_hash = 0;
    board->non_pawn_hash[0] = 0;
    board->non_pawn_hash[1] = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int p = board->board[sq];
        if (p != EMPTY) {
            int col = board->colors[sq];
            if (p == PAWN) board->pawn_hash ^= piece_keys[PAWN * 2 + col][sq];
            else board->non_pawn_hash[col] ^= piece_keys[p * 2 + col][sq];
        }
    }
}

bool make_move(Board* board, Move move, int ply, int thread_id) {
    (void)thread_id;
    if (ply < 0 || ply >= MAX_PLY || move == MOVE_NONE) return false;

    int from = move_from(move);
    int to = move_to(move);
    if (from == to) return false;

    int piece = board->board[from];
    int color = board->colors[from];
    if (piece == EMPTY || color != board->turn) return false;

    int captured = move_captured(move);
    int promo = move_promo(move);
    bool is_ep = move_is_ep(move);
    bool is_castle = move_is_castle(move);

    board->history_hash[ply] = board->hash;
    board->history_pawn_hash[ply] = board->pawn_hash;
    board->history_non_pawn_hash[ply][0] = board->non_pawn_hash[0];
    board->history_non_pawn_hash[ply][1] = board->non_pawn_hash[1];
    board->history_castling[ply] = board->castling_rights;
    board->history_en_passant[ply] = board->en_passant_square;
    board->history_halfmove[ply] = board->halfmove_clock;

    int old_castling = board->castling_rights;
    int new_castling = old_castling;
    new_castling &= ~((from == E1 || from == H1 || to == H1) ? WK_CASTLE : 0);
    new_castling &= ~((from == E1 || from == A1 || to == A1) ? WQ_CASTLE : 0);
    new_castling &= ~((from == E8 || from == H8 || to == H8) ? BK_CASTLE : 0);
    new_castling &= ~((from == E8 || from == A8 || to == A8) ? BQ_CASTLE : 0);

    int p_idx = (color == WHITE ? 0 : 6) + piece - 1;
    U64 from_to_mask = (1ULL << from) | (1ULL << to);

    board->pieces[p_idx] ^= from_to_mask;
    board->occupied[color] ^= from_to_mask;

    board->board[from] = EMPTY;
    board->colors[from] = -1;
    board->board[to] = piece;
    board->colors[to] = color;

    U64 k_move = piece_keys[piece * 2 + color][from] ^ piece_keys[piece * 2 + color][to];
    board->hash ^= k_move;
    if (piece == PAWN) board->pawn_hash ^= k_move;
    else board->non_pawn_hash[color] ^= k_move;

    if (captured != EMPTY && !is_ep) {
        int cap_idx = (1 - color == WHITE ? 0 : 6) + captured - 1;
        board->pieces[cap_idx] ^= (1ULL << to);
        board->occupied[1 - color] ^= (1ULL << to);
        U64 k_cap = piece_keys[captured * 2 + (1 - color)][to];
        board->hash ^= k_cap;
        if (captured == PAWN) board->pawn_hash ^= k_cap;
        else board->non_pawn_hash[1 - color] ^= k_cap;
    }

    if (is_ep) {
        int cap_sq = to + ((color == WHITE) ? -8 : 8);
        int cap_idx = (1 - color == WHITE ? 0 : 6) + PAWN - 1;
        board->pieces[cap_idx] ^= (1ULL << cap_sq);
        board->occupied[1 - color] ^= (1ULL << cap_sq);
        board->board[cap_sq] = EMPTY;
        board->colors[cap_sq] = -1;
        U64 k_ep = piece_keys[PAWN * 2 + (1 - color)][cap_sq];
        board->hash ^= k_ep;
        board->pawn_hash ^= k_ep;
    }

    if (is_castle) {
        int r_from = -1, r_to = -1;
        switch (to) {
            case G1: r_from = H1; r_to = F1; break;
            case C1: r_from = A1; r_to = D1; break;
            case G8: r_from = H8; r_to = F8; break;
            case C8: r_from = A8; r_to = D8; break;
        }
        int rook_idx = (color == WHITE ? 0 : 6) + ROOK - 1;
        U64 r_mask = (1ULL << r_from) | (1ULL << r_to);
        board->pieces[rook_idx] ^= r_mask;
        board->occupied[color] ^= r_mask;
        board->board[r_from] = EMPTY;
        board->colors[r_from] = -1;
        board->board[r_to] = ROOK;
        board->colors[r_to] = color;
        U64 k_rook = piece_keys[ROOK * 2 + color][r_from] ^ piece_keys[ROOK * 2 + color][r_to];
        board->hash ^= k_rook;
        board->non_pawn_hash[color] ^= k_rook;
    }

    if (promo != EMPTY) {
        board->pieces[p_idx] ^= (1ULL << to);
        int promo_idx = (color == WHITE ? 0 : 6) + promo - 1;
        board->pieces[promo_idx] |= (1ULL << to);
        board->board[to] = promo;
        U64 k_p = piece_keys[PAWN * 2 + color][to];
        U64 k_pr = piece_keys[promo * 2 + color][to];
        board->hash ^= k_p ^ k_pr;
        board->pawn_hash ^= k_p;
        board->non_pawn_hash[color] ^= k_pr;
    }

    board->occupied[2] = board->occupied[0] | board->occupied[1];

    if (piece == KING) board->king_sq[color] = to;

    board->en_passant_square = NO_SQ;
    if (piece == PAWN && abs(from - to) == 16) {
        board->en_passant_square = from + (to - from) / 2;
    }

    if (board->history_en_passant[ply] != NO_SQ) board->hash ^= ep_keys[board->history_en_passant[ply]];
    if (board->en_passant_square != NO_SQ) board->hash ^= ep_keys[board->en_passant_square];

    board->castling_rights = new_castling;
    board->hash ^= castle_keys[old_castling];
    board->hash ^= castle_keys[new_castling];

    board->turn = 1 - board->turn;
    board->hash ^= side_key;

    if (piece == PAWN || captured != EMPTY || is_ep) {
        board->halfmove_clock = 0;
    } else {
        board->halfmove_clock++;
    }

    if (is_square_attacked(board, board->king_sq[color], board->turn)) {
        unmake_move(board, move, ply);
        return false;
    }
    return true;
}

void unmake_move(Board* board, Move move, int ply) {
    board->turn = 1 - board->turn;
    int from = move_from(move);
    int to = move_to(move);
    int promo = move_promo(move);
    int captured = move_captured(move);
    bool is_ep = move_is_ep(move);
    bool is_castle = move_is_castle(move);

    int color = board->turn;
    int piece = (promo != EMPTY) ? PAWN : board->board[to];

    int p_idx = (color == WHITE ? 0 : 6) + piece - 1;
    U64 from_to_mask = (1ULL << from) | (1ULL << to);

    if (promo != EMPTY) {
        int promo_idx = (color == WHITE ? 0 : 6) + promo - 1;
        board->pieces[promo_idx] ^= (1ULL << to);
        board->pieces[p_idx] |= (1ULL << from);
    } else {
        board->pieces[p_idx] ^= from_to_mask;
    }

    board->occupied[color] ^= from_to_mask;
    board->board[from] = piece;
    board->colors[from] = color;
    board->board[to] = captured;
    board->colors[to] = (captured != EMPTY) ? (1 - color) : -1;

    if (captured != EMPTY && !is_ep) {
        int cap_idx = (1 - color == WHITE ? 0 : 6) + captured - 1;
        board->pieces[cap_idx] |= (1ULL << to);
        board->occupied[1 - color] |= (1ULL << to);
    }

    if (is_ep) {
        int cap_sq = to + ((color == WHITE) ? -8 : 8);
        int cap_idx = (1 - color == WHITE ? 0 : 6) + PAWN - 1;
        board->pieces[cap_idx] ^= (1ULL << cap_sq);
        board->occupied[1 - color] ^= (1ULL << cap_sq);
        board->board[cap_sq] = PAWN;
        board->colors[cap_sq] = 1 - color;
        board->board[to] = EMPTY;
        board->colors[to] = -1;
    }

    if (is_castle) {
        int r_from = -1, r_to = -1;
        switch (to) {
            case G1: r_from = H1; r_to = F1; break;
            case C1: r_from = A1; r_to = D1; break;
            case G8: r_from = H8; r_to = F8; break;
            case C8: r_from = A8; r_to = D8; break;
        }
        int rook_idx = (color == WHITE ? 0 : 6) + ROOK - 1;
        U64 r_mask = (1ULL << r_from) | (1ULL << r_to);
        board->pieces[rook_idx] ^= r_mask;
        board->occupied[color] ^= r_mask;
        board->board[r_from] = ROOK;
        board->colors[r_from] = color;
        board->board[r_to] = EMPTY;
        board->colors[r_to] = -1;
    }

    board->occupied[2] = board->occupied[0] | board->occupied[1];

    if (piece == KING) board->king_sq[color] = from;

    board->castling_rights = board->history_castling[ply];
    board->en_passant_square = board->history_en_passant[ply];
    board->halfmove_clock = board->history_halfmove[ply];
    board->hash = board->history_hash[ply];
    board->pawn_hash = board->history_pawn_hash[ply];
    board->non_pawn_hash[0] = board->history_non_pawn_hash[ply][0];
    board->non_pawn_hash[1] = board->history_non_pawn_hash[ply][1];
}