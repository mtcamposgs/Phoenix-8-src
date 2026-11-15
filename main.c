#include "defs.h"

int main(void) {
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);

    init_zobrist();
    init_bitboards();
    init_search_tables();

    if (!init_shra_embedded()) {
        fprintf(stderr, "Failed to initialize embedded evaluation network.\n");
        return EXIT_FAILURE;
    }

    printf("%s by %s\n", ENGINE_NAME, ENGINE_AUTHOR);
    uci_loop();
    return 0;
}