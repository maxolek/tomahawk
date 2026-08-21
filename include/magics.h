#ifndef MAGICS_H
#define MAGICS_H

#include "helpers.h"
#include <immintrin.h>

namespace Magics {

    struct Magic {
        U64 mask;

        // Offset into the global packed attack table.
        uint32_t offset;

        // PEXT index -> compact attack-table index.
        // Max PEXT index is 4095 for rooks, so uint16_t is sufficient.
        uint16_t* indexMap;

        uint16_t pextSize;
        uint16_t attackSize;
    };

    extern Magic rook[64];
    extern Magic bishop[64];

    extern std::vector<U64> rookAttackTable;
    extern std::vector<U64> bishopAttackTable;

    void initMagics();

    U64 rookAttacks(int sq, U64 occ);
    U64 bishopAttacks(int sq, U64 occ);
}

#endif