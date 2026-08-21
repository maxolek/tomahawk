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

    // Runtime lookup
    inline U64 rookAttacks(int sq, U64 occ) {
        const Magic& magic = rook[sq];
        return rookAttackTable[magic.offset + magic.indexMap[_pext_u64(occ, magic.mask)]];
    }

    inline U64 bishopAttacks(int sq, U64 occ) {
        const Magic& magic = bishop[sq];
        return bishopAttackTable[magic.offset + magic.indexMap[_pext_u64(occ, magic.mask)]];
    }
}

#endif