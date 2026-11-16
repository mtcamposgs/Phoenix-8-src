#include "defs.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <io.h>

uint64_t get_time_ms(void) {
    return (uint64_t)GetTickCount64();
}

int input_waiting(void) {
    static int init = 0, pipe = 0;
    static HANDLE inh;
    DWORD dw = 0;
    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
    }
    if (pipe) {
        if (!PeekNamedPipe(inh, NULL, 0, NULL, &dw, NULL)) return 0;
        return dw > 0;
    }
    return 0;
}
#else
#include <sys/time.h>
#include <sys/select.h>
#include <unistd.h>

uint64_t get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec * 1000ULL + tv.tv_usec / 1000ULL);
}

int input_waiting(void) {
    fd_set readfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &readfds);
}
#endif

int g_move_overhead = DEFAULT_MOVE_OVERHEAD;

static int parse_square(const char* sq_str) {
    if (!sq_str || sq_str[0] < 'a' || sq_str[0] > 'h' ||
        sq_str[1] < '1' || sq_str[1] > '8') return NO_SQ;
    return (sq_str[0] - 'a') + (sq_str[1] - '1') * 8;
}

static bool parse_move(Board* board, const char* move_str, Move* out_move) {
    if (!board || !move_str || !out_move) return false;
    size_t len = strlen(move_str);
    if (len != 4 && len != 5) return false;

    int from = parse_square(move_str);
    int to = parse_square(move_str + 2);
    if (from == NO_SQ || to == NO_SQ || from == to) return false;

    int promo_piece = EMPTY;
    if (len == 5) {
        switch (move_str[4]) {
            case 'q': promo_piece = QUEEN; break;
            case 'r': promo_piece = ROOK; break;
            case 'b': promo_piece = BISHOP; break;
            case 'n': promo_piece = KNIGHT; break;
            default: return false;
        }
    }

    MoveList list;
    generate_moves(board, &list);
    for (int i = 0; i < list.count; ++i) {
        Move candidate = list.moves[i];
        if (move_from(candidate) == from && move_to(candidate) == to &&
            move_promo(candidate) == promo_piece) {
            *out_move = candidate;
            return true;
        }
    }
    return false;
}

static uint64_t calculate_time_allocated(const Board* board, const SearchInfo* info, const char* line) {
    int overhead = g_move_overhead;
    if (info->movetime > 0) {
        long t = (long)info->movetime - overhead;
        return (t > 10) ? (uint64_t)t : 10;
    }

    if (strstr(line, "infinite")) return 1000000000ULL;

    int time_left = (board->turn == WHITE) ? info->wtime : info->btime;
    int increment = (board->turn == WHITE) ? info->winc : info->binc;

    if (time_left <= 0 && info->movetime <= 0 && info->depth == MAX_PLY) return 3000;
    if (time_left <= overhead + 50) return 20;

    int moves_remaining;
    if (info->movestogo > 0) {
        moves_remaining = info->movestogo;
    } else {
        int current_move_num = board->ply_count / 2;
        if (increment == 0) {
            moves_remaining = 45 - (current_move_num / 3);
            if (moves_remaining < 25) moves_remaining = 25;
        } else {
            moves_remaining = 35 - (current_move_num / 4);
            if (moves_remaining < 20) moves_remaining = 20;
        }
    }

    double usable_time = (double)(time_left - overhead);
    double base_time = usable_time / (double)moves_remaining;
    double time_for_move_ms = base_time + (increment * 0.75);

    double max_ratio = (info->movestogo == 1) ? 0.55 : ((info->movestogo > 1) ? 0.35 : 0.20);
    double max_time_ms = usable_time * max_ratio;
    if (time_for_move_ms > max_time_ms) time_for_move_ms = max_time_ms;

    return (time_for_move_ms > 20.0) ? (uint64_t)time_for_move_ms : 20;
}

static void trim_newline(char* s) {
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[--len] = '\0';
    }
}

static void run_bench(int bench_depth) {
    static const char* bench_fens[6] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "4k3/8/8/4b3/8/4K3/p7/7Q w - - 0 1"
    };

    if (bench_depth < 1) bench_depth = 1;
    if (bench_depth > 20) bench_depth = 20;

    Board b;
    uint64_t t0 = get_time_ms();
    clear_tt();
    clear_search_history();

    for (int i = 0; i < 6; ++i) {
        parse_fen(&b, bench_fens[i]);
        b.game_ply = 0;
        SearchInfo info = {0};
        info.depth = bench_depth;
        info.start_time = get_time_ms();
        info.stop_time = info.start_time + 1000000000ULL;
        new_search_tt();
        printf("Bench position %d/6 (depth %d):\n", i + 1, bench_depth);
        search_root(&b, &info);
    }

    uint64_t dt = get_time_ms() - t0;
    printf("Bench finished in %llu ms.\n", (unsigned long long)dt);
    fflush(stdout);
}

void uci_loop(void) {
    static char line[65536];
    Board board;
    bool position_valid = true;

    reset_board(&board);
    init_tt(DEFAULT_TT_SIZE_MB);

    while (fgets(line, sizeof(line), stdin)) {
        trim_newline(line);
        if (line[0] == '\0') continue;

        if (strcmp(line, "ucinewgame") == 0) {
            clear_tt();
            clear_search_history();
            reset_board(&board);
            position_valid = true;
        } else if (strcmp(line, "uci") == 0) {
            printf("id name %s\n", ENGINE_NAME);
            printf("id author %s\n", ENGINE_AUTHOR);
            printf("option name Threads type spin default 1 min 1 max %d\n", MAX_THREADS);
            printf("option name Hash type spin default %d min 1 max 16384\n", DEFAULT_TT_SIZE_MB);
            printf("option name Clear Hash type button\n");
            printf("option name Move Overhead type spin default %d min 0 max 5000\n", DEFAULT_MOVE_OVERHEAD);
            printf("option name EvalFile type string default <internal>\n");
            printf("uciok\n");
            fflush(stdout);
        } else if (strcmp(line, "isready") == 0) {
            printf("readyok\n");
            fflush(stdout);
        } else if (strcmp(line, "bench") == 0 || strncmp(line, "bench ", 6) == 0) {
            int bd = 10;
            if (strncmp(line, "bench ", 6) == 0) {
                sscanf(line + 6, "%d", &bd);
            }
            run_bench(bd);
            reset_board(&board);
            position_valid = true;
        } else if (strncmp(line, "setoption ", 10) == 0) {
            char* name_ptr = strstr(line, "name ");
            if (!name_ptr) continue;
            name_ptr += 5;
            while (*name_ptr == ' ') name_ptr++;

            char* val_ptr = strstr(name_ptr, " value ");
            char opt_name[128] = {0};
            const char* opt_val = NULL;

            if (val_ptr) {
                size_t nlen = (size_t)(val_ptr - name_ptr);
                while (nlen > 0 && name_ptr[nlen - 1] == ' ') nlen--;
                if (nlen >= sizeof(opt_name)) nlen = sizeof(opt_name) - 1;
                memcpy(opt_name, name_ptr, nlen);
                opt_name[nlen] = '\0';

                opt_val = val_ptr + 7;
                while (*opt_val == ' ') opt_val++;
            } else {
                strncpy(opt_name, name_ptr, sizeof(opt_name) - 1);
                size_t nlen = strlen(opt_name);
                while (nlen > 0 && opt_name[nlen - 1] == ' ') opt_name[--nlen] = '\0';
            }

            if (strcmp(opt_name, "Threads") == 0 && opt_val) {
                int th = atoi(opt_val);
                if (th < 1) th = 1;
                if (th > MAX_THREADS) th = MAX_THREADS;
                g_num_threads = th;
            } else if (strcmp(opt_name, "Hash") == 0 && opt_val) {
                int mb = atoi(opt_val);
                init_tt(mb);
            } else if (strcmp(opt_name, "Clear Hash") == 0) {
                clear_tt();
            } else if (strcmp(opt_name, "Move Overhead") == 0 && opt_val) {
                int mo = atoi(opt_val);
                if (mo < 0) mo = 0;
                if (mo > 5000) mo = 5000;
                g_move_overhead = mo;
            } else if (strcmp(opt_name, "EvalFile") == 0 && opt_val) {
                if (!init_shra(opt_val)) {
                    printf("info string Failed to load EvalFile: %s\n", opt_val);
                    fflush(stdout);
                }
            }
        } else if (strncmp(line, "position ", 9) == 0) {
            position_valid = true;
            char* ptr = line + 9;
            while (*ptr == ' ') ptr++;

            char* moves_ptr = strstr(ptr, " moves");
            if (moves_ptr) {
                *moves_ptr = '\0';
                moves_ptr += 6;
            }

            if (strncmp(ptr, "startpos", 8) == 0) {
                reset_board(&board);
            } else if (strncmp(ptr, "fen ", 4) == 0) {
                ptr += 4;
                while (*ptr == ' ') ptr++;
                parse_fen(&board, ptr);
            }

            board.game_ply = 0;

            if (moves_ptr) {
                char* p = moves_ptr;
                while (*p) {
                    while (*p == ' ' || *p == '\t') p++;
                    if (!*p) break;

                    char* move_str = p;
                    while (*p && *p != ' ' && *p != '\t') p++;
                    if (*p) *p++ = '\0';

                    Move move;
                    if (!parse_move(&board, move_str, &move)) {
                        position_valid = false;
                        break;
                    }
                    U64 previous_hash = board.hash;
                    if (!make_move(&board, move, 0, 0)) {
                        position_valid = false;
                        break;
                    }
                    if (board.game_ply < 2048) {
                        board.game_history[board.game_ply++] = previous_hash;
                    }
                    board.ply_count++;
                }
            }
        } else if (strcmp(line, "go") == 0 || strncmp(line, "go ", 3) == 0) {
            if (!position_valid) {
                printf("bestmove 0000\n");
                fflush(stdout);
                continue;
            }
            SearchInfo info = {0};
            info.depth = MAX_PLY;

            char* token;
            bool depth_command_given = false;

            if ((token = strstr(line, "wtime")))     sscanf(token, "wtime %d", &info.wtime);
            if ((token = strstr(line, "btime")))     sscanf(token, "btime %d", &info.btime);
            if ((token = strstr(line, "winc")))      sscanf(token, "winc %d", &info.winc);
            if ((token = strstr(line, "binc")))      sscanf(token, "binc %d", &info.binc);
            if ((token = strstr(line, "movestogo"))) sscanf(token, "movestogo %d", &info.movestogo);
            if ((token = strstr(line, "movetime")))  sscanf(token, "movetime %d", &info.movetime);

            if ((token = strstr(line, "nodes"))) {
                unsigned long long n = 0;
                if (sscanf(token, "nodes %llu", &n) == 1) {
                    info.max_nodes = n;
                }
            }

            if ((token = strstr(line, "depth"))) {
                sscanf(token, "depth %d", &info.depth);
                depth_command_given = true;
            }

            info.start_time = get_time_ms();

            if (info.max_nodes > 0 && info.wtime <= 0 && info.btime <= 0 && info.movetime <= 0) {
                info.stop_time = info.start_time + 1000000000ULL;
            } else if (depth_command_given && info.wtime <= 0 && info.btime <= 0 && info.movetime <= 0) {
                info.stop_time = info.start_time + 1000000000ULL;
            } else {
                uint64_t allocated = calculate_time_allocated(&board, &info, line);
                info.stop_time = info.start_time + allocated;
            }

            new_search_tt();

            Move best_move = search_root(&board, &info);

            if (best_move == MOVE_NONE) {
                printf("bestmove 0000\n");
            } else {
                int from = move_from(best_move);
                int to = move_to(best_move);
                int promo = move_promo(best_move);
                char promo_char = '\0';
                if (promo == QUEEN) promo_char = 'q';
                else if (promo == ROOK) promo_char = 'r';
                else if (promo == BISHOP) promo_char = 'b';
                else if (promo == KNIGHT) promo_char = 'n';

                if (promo_char != '\0') {
                    printf("bestmove %c%c%c%c%c\n",
                        'a' + (from % 8), '1' + (from / 8),
                        'a' + (to % 8),   '1' + (to / 8), promo_char);
                } else {
                    printf("bestmove %c%c%c%c\n",
                        'a' + (from % 8), '1' + (from / 8),
                        'a' + (to % 8),   '1' + (to / 8));
                }
            }
            fflush(stdout);
        } else if (strcmp(line, "quit") == 0) {
            break;
        }
    }
}
