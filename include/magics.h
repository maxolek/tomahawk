#ifndef MAGICS_H
#define MAGICS_H

#include "helpers.h"
#ifdef _WIN32
    #include <immintrin.h>
#endif

namespace Magics {

#ifdef _WIN32
    struct PextTable {
        U64 mask;
        uint32_t offset;
    };

    extern PextTable rookPext[64];
    extern PextTable bishopPext[64];

    void buildPEXTTable(
        PextTable& table,
        std::vector<U64>& attackTable,
        int sq,
        bool rookPiece
    );

    //void buildCompactTable(
    //    Magic& magic,
    //    std::vector<U64>& globalTable,
    //    int sq,
    //    bool rookPiece
    //);
    
    extern std::vector<U64> rookAttackTable;
    extern std::vector<U64> bishopAttackTable;
    
    void initPEXTMagics();

    // Runtime lookup
    inline U64 rookAttacks(int sq, U64 occ) {
        const PextTable& t = rookPext[sq]; // rook[sq];
        return rookAttackTable[t.offset + _pext_u64(occ, t.mask)];
    }
    
    inline U64 bishopAttacks(int sq, U64 occ) {
        const PextTable& t = bishopPext[sq]; // bishop[sq];
        return bishopAttackTable[t.offset + _pext_u64(occ, t.mask)];
    }
#else
    extern U64 rookAttackTable[64][4096]; // precompute and store
    extern U64 bishopAttackTable[64][512];
    extern U64 rookMasks[64];
    extern U64 bishopMasks[64];
    extern U64 rookMagics[64];
    extern U64 bishopMagics[64];
    extern int rookShifts[64];
    extern int bishopShifts[64];

    void initMagics(); // call at engine startup
    U64 maskRook(int sq);
    U64 maskBishop(int sq);
    std::vector<U64> generateAllOccupancies(U64 mask);
    U64 rookAttacksOnTheFly(int sq, U64 blockers);
    U64 bishopAttacksOnTheFly(int sq, U64 blockers);
    U64 rookAttacks(int sq, U64 occ);
    U64 bishopAttacks(int sq, U64 occ);
#endif
}

#endif