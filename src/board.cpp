// Board class: represents the current state of the chess board and provides
// functions for move execution, undo, bitboards, Zobrist hashing, and FEN handling.

// Includes
#include <PrecomputedMoveData.h>
#include <move.h>
#include <gamestate.h>
#include <board.h>
#include <magics.h>

// ------------------------------------------------------------
// Constructors
// ------------------------------------------------------------

Board::Board(std::string _fen) {
    setFromFEN(_fen);
}

// deep copy
Board::Board(const Board& other) {
    memcpy(pieceBitboards, other.pieceBitboards, sizeof(pieceBitboards));
    memcpy(colorBitboards, other.colorBitboards, sizeof(colorBitboards));
    memcpy(sqToPiece, other.sqToPiece, sizeof(sqToPiece));
    currentGameState = other.currentGameState;
    zobrist_hash = other.zobrist_hash;
    is_white_move = other.is_white_move;
    move_color = other.move_color;
    allGameMoves = other.allGameMoves;       // deep copy
    hash_history = other.hash_history;       // deep copy
    zobrist_history = other.zobrist_history; // deep copy
    gameStateHistory = other.gameStateHistory;

    //auditZobrist(other, "post-copy");
    //assert(zobrist_hash == computeZobristHash());
}


// ------------------------------------------------------------
// Move execution / undo
// ------------------------------------------------------------
void Board::MakeMove(Move move) {
    #ifdef DEV
        ScopedTimer timer(T_MAKEMOVE);
    #endif

    //currentGameState.was_in_check = is_in_check;

    int old_castling = currentGameState.castlingRights;
    int oldEp = currentGameState.enPassantFile;
    if (oldEp > -1) zobrist_hash ^= zobrist_enpassant[oldEp];

    int start_square = move.StartSquare();
    int target_square = move.TargetSquare();
    int move_flag = move.MoveFlag();
    bool is_promotion = move.IsPromotion();
    bool is_enpassant = (move_flag == Move::enPassantCaptureFlag);

    int moved_piece = getMovedPiece(start_square);
    int captured_piece = is_enpassant ? 0 : getCapturedPiece(target_square);
    int promotion_piece = move.PromotionPieceType();

    MovePiece(moved_piece, start_square, target_square);

    if (captured_piece > -1)
        CapturePiece(captured_piece, target_square, is_enpassant, moved_piece == captured_piece);

    // Handle castling
    if (move_flag == Move::castleFlag) {
        if (!move_color) {
            if (target_square == g1) MovePiece(rook, h1, f1);
            else MovePiece(rook, a1, d1);
            white_castled = true;
        } else {
            if (target_square == g8) MovePiece(rook, h8, f8);
            else MovePiece(rook, a8, d8);
            black_castled = true;
        }
    }

    // Promotion
    if (is_promotion) PromoteToPiece(promotion_piece, target_square);

    // Update game state
    currentGameState.capturedPieceType = captured_piece;

    // --------------- SET NEW EP FILE --------------------
    int newEp = -1;
    if (move_flag == Move::pawnTwoUpFlag)
        newEp = start_square & 7;   // file of the pawn
    currentGameState.enPassantFile = newEp;
    // --------------- ADD NEW EP HASH --------------------
    if (newEp != -1 ) {
        zobrist_hash ^= zobrist_enpassant[newEp];
    }

    updateFiftyMoveCounter(moved_piece, captured_piece > -1);

    // Castling rights
    if (currentGameState.castlingRights != 0) {
        if (moved_piece == king) {
            if (is_white_move) {
                currentGameState.castlingRights &= GameState::clearWhiteKingSideMask;
                currentGameState.castlingRights &= GameState::clearWhiteQueenSideMask;
            } else {
                currentGameState.castlingRights &= GameState::clearBlackKingSideMask;
                currentGameState.castlingRights &= GameState::clearBlackQueenSideMask;
            }
        } else if (moved_piece == rook) {
            switch (start_square) {
                case a1: currentGameState.castlingRights &= GameState::clearWhiteQueenSideMask; break;
                case h1: currentGameState.castlingRights &= GameState::clearWhiteKingSideMask; break;
                case a8: currentGameState.castlingRights &= GameState::clearBlackQueenSideMask; break;
                case h8: currentGameState.castlingRights &= GameState::clearBlackKingSideMask; break;
            }
        }
        if (captured_piece == rook) {
            switch (target_square) {
                case a1: currentGameState.castlingRights &= GameState::clearWhiteQueenSideMask; break;
                case h1: currentGameState.castlingRights &= GameState::clearWhiteKingSideMask; break;
                case a8: currentGameState.castlingRights &= GameState::clearBlackQueenSideMask; break;
                case h8: currentGameState.castlingRights &= GameState::clearBlackKingSideMask; break;
            }
        }
    }

    zobrist_hash ^= zobristCastlingHash(old_castling);
    zobrist_hash ^= zobristCastlingHash(currentGameState.castlingRights);

    is_in_check = inCheck(false);
    plyCount++;
    allGameMoves.push_back(move);

    zobrist_hash ^= zobrist_side_to_move;
    is_white_move = !is_white_move;
    move_color = 1 - move_color;

    hash_history[zobrist_hash]++;
    zobrist_history.push_back(zobrist_hash);
    gameStateHistory.push_back(currentGameState);
}

void Board::UnmakeMove(Move move) {
    #ifdef DEV
        ScopedTimer timer(T_UNMAKE_MOVE);
    #endif

    auto it = hash_history.find(zobrist_hash);
    if (it != hash_history.end()) {
        it->second--;
        if (it->second <= 0)
            hash_history.erase(it);
        zobrist_history.pop_back();
    } //else {
        // Optional: error check
        //std::cerr << "UnmakeMove: hash not found in history!\n";
        // Maybe abort or handle gracefully
    //}

    //if (hash_history[zobrist_hash] < 0) {std::cout << "tf" << std::endl; exit(0);}

    int new_castling = currentGameState.castlingRights;

    // Side to move
    zobrist_hash ^= zobrist_side_to_move;
    is_white_move = !is_white_move;
    move_color = 1 - move_color;

    int moved_from = move.StartSquare();
    int moved_to = move.TargetSquare();
    int move_flag = move.MoveFlag();
    int moved_piece = (move.IsPromotion() || move_flag == Move::enPassantCaptureFlag || move_flag == Move::pawnTwoUpFlag)
                      ? pawn : getMovedPiece(moved_to);
    int captured_piece = currentGameState.capturedPieceType;
    int promotion_piece = move.PromotionPieceType();

    // restore pawn_endgame flag for NMP
    if (pawn_endgame && (captured_piece > pawn && captured_piece < king)) {
        pawn_endgame = false;
    }

    // --- Remove old EP hash (set by move being undone) ---
    if (currentGameState.enPassantFile != -1) // note: opposite side
    {
        zobrist_hash ^= zobrist_enpassant[currentGameState.enPassantFile];
    }

    // --- Move piece back ---
    MovePiece(moved_piece, moved_to, moved_from);

    // --- Undo promotion ---
    if (move.IsPromotion()) {
        pop_bit(pieceBitboards[promotion_piece], moved_to);
        zobrist_hash ^= zobrist_table[move_color*6 + promotion_piece][moved_to];
        zobrist_hash ^= zobrist_table[move_color*6 + pawn][moved_to]; // movePiece() resets so it must be undone
    }

    // --- Restore captured piece ---
    if (move_flag == Move::enPassantCaptureFlag) {
        int ep_square = moved_to + ((move_color == 0) ? -8 : 8);
        set_bit(pieceBitboards[pawn], ep_square);
        set_bit(colorBitboards[1 - move_color], ep_square);
        putPiece(pawn + (1 - move_color) * 6, ep_square);
        zobrist_hash ^= zobrist_table[(1-move_color)*6+ pawn][ep_square];
    } else if (captured_piece > -1) {
        set_bit(pieceBitboards[captured_piece], moved_to);
        set_bit(colorBitboards[1 - move_color], moved_to);
        putPiece(captured_piece + (1 - move_color) * 6, moved_to);
        zobrist_hash ^= zobrist_table[(1-move_color)*6 + captured_piece][moved_to];
    }

    // --- Undo castling rook moves ---
    if (move_flag == Move::castleFlag) {
        if (moved_to == g1) MovePiece(rook, f1, h1), white_castled = false;
        else if (moved_to == c1) MovePiece(rook, d1, a1), white_castled = false;
        else if (moved_to == g8) MovePiece(rook, f8, h8), black_castled = false;
        else if (moved_to == c8) MovePiece(rook, d8, a8), black_castled = false;
    }

    // --- Restore previous game state ---
    is_in_check = inCheck(true);
    gameStateHistory.pop_back();
    currentGameState = gameStateHistory.back();
    //is_in_check = currentGameState.was_in_check;
    

    // --- Castling hash ---
    zobrist_hash ^= zobristCastlingHash(new_castling);
    zobrist_hash ^= zobristCastlingHash(currentGameState.castlingRights);

    // --- Restore EP hash (the EP square before the move) ---
    if (currentGameState.enPassantFile != -1)
    {
        zobrist_hash ^= zobrist_enpassant[currentGameState.enPassantFile];
    }

    // --- Finish ---
    plyCount--;
    allGameMoves.pop_back();
}

// -----------------------------------
// Null Moves
// -----------------------------------

// minimal implementations of moves
// no hash updates, move history, or game state updates since these are not needed 
// for null move pruning and would be expensive to maintain
void Board::MakeNullMove() {
    //ScopedTimer timer(T_MAKENULLMOVE);

    // clear EP + captured_piece
    currentGameState.capturedPieceType = -1;
    if (currentGameState.enPassantFile != -1) {
        zobrist_hash ^= zobrist_enpassant[currentGameState.enPassantFile];
        currentGameState.enPassantFile = -1;
    }

    // flipp side to move
    zobrist_hash ^= zobrist_side_to_move;
    is_white_move = !is_white_move;
    move_color = 1 - move_color;

    // track history for NMP-tree TT
    // wont plague regular TT since STM is flipped so hash's will never match
    gameStateHistory.push_back(currentGameState);
    //hash_history[zobrist_hash]++;
    //zobrist_history.push_back(zobrist_hash);
    allGameMoves.push_back(Move::NullMove());
    //gameStateHistory.push_back(currentGameState);

    is_in_check = false;
    plyCount++;
}

void Board::UnmakeNullMove() {
    //ScopedTimer timer(T_UNMAKENULLMOVE);

    // undo hash history
    //auto it = hash_history.find(zobrist_hash);
    //if (it != hash_history.end()) {
    //    it->second--;
    //   if (it->second <= 0)
    //        hash_history.erase(it);
    //}
    //zobrist_history.pop_back();

    // flip STM back
    zobrist_hash ^= zobrist_side_to_move;
    is_white_move = !is_white_move;
    move_color = 1 - move_color;

    // restore gamestate and re-add EP hash if the original state had one
    allGameMoves.pop_back();
    gameStateHistory.pop_back(); // post-null-move gameState
    currentGameState = gameStateHistory.back(); // gameState after last real move
    if (currentGameState.enPassantFile != -1)
        zobrist_hash ^= zobrist_enpassant[currentGameState.enPassantFile];

    is_in_check = false;
    plyCount--;
}

// ------------------------------------------------------------
// Bitboard manipulation helpers
// ------------------------------------------------------------
void Board::putPiece(int pt12, int sq) { sqToPiece[sq] = pt12; }
void Board::removePiece(int sq) {
    int pt12 = sqToPiece[sq];
    if (pt12 == -1) return;
    sqToPiece[sq] = -1;
}

void Board::MovePiece(int piece, int start_square, int target_square) {
    if (piece == -1) return;

    set_bit(pieceBitboards[piece], target_square);
    pop_bit(pieceBitboards[piece], start_square);
    set_bit(colorBitboards[move_color], target_square);
    pop_bit(colorBitboards[move_color], start_square);

    removePiece(start_square);
    putPiece(piece + (is_white_move ? 0 : 6), target_square);

    // b-p =0, w-p =1, b-n=2, w-n=3, etc..
    zobrist_hash ^= zobrist_table[move_color*6 + piece][start_square]; //[move_color*6 + piece][start_square];
    zobrist_hash ^= zobrist_table[move_color*6 + piece][target_square];
}

void Board::CapturePiece(int piece, int target_square, bool is_enpassant, bool captured_is_moved_piece) {
    if (is_enpassant) {
        int cap_sq = (move_color == 0) ? target_square - 8 : target_square + 8;

        pop_bit(pieceBitboards[piece], cap_sq);
        pop_bit(colorBitboards[1-move_color], cap_sq);
        zobrist_hash ^= zobrist_table[(1-move_color)*6 + piece][int(cap_sq)];
    } else if (captured_is_moved_piece) {
        pop_bit(colorBitboards[1-move_color], target_square);
        zobrist_hash ^= zobrist_table[(1-move_color)*6 + piece][target_square];
    } else {
        pop_bit(pieceBitboards[piece], target_square);
        pop_bit(colorBitboards[1-move_color], target_square);
        zobrist_hash ^= zobrist_table[(1-move_color)*6 + piece][target_square];
    }

    if (
        !pawn_endgame
        && !(pieceBitboards[knight]
                | pieceBitboards[bishop]
                | pieceBitboards[rook]
                | pieceBitboards[queen])
    ) {
        pawn_endgame = true;
    }
}

void Board::PromoteToPiece(int piece, int target_square) {
    pop_bit(pieceBitboards[pawn], target_square);
    set_bit(pieceBitboards[piece], target_square);
    zobrist_hash ^= zobrist_table[move_color*6 + piece][target_square];
    zobrist_hash ^= zobrist_table[move_color*6 + pawn][target_square];
}

// ------------------------------------------------------------
// Piece / square queries
// ------------------------------------------------------------
int Board::getMovedPiece(int start_square) const {
    for (int piece=0; piece<6; piece++)
        if (get_bit(pieceBitboards[piece], start_square))
            return piece;
    return -1;
}

int Board::getCapturedPiece(int target_square) const {
    for (int piece=0; piece<6; piece++)
        if (get_bit(pieceBitboards[piece], target_square))
            return piece;
    return -1;
}

int Board::getSideAt(int square) const {
    for (int side=0; side<2; side++)
        if (colorBitboards[side] & (1ULL << square))
            return side;
    return -1;
}

int Board::getPieceAt(int square, int side) const {
    for (int piece=0; piece<6; piece++)
        if (get_bit(pieceBitboards[piece], square))
            return piece + side*10;
    return -1;
}

int Board::kingSquare(bool white) const {
    U64 kings = pieceBitboards[king];
    U64 side = white ? colorBitboards[0] : colorBitboards[1];
    return sqidx(kings & side);
}

bool Board::canEnpassantCapture(int epFile) const {
    if (epFile < 0 || epFile > 7) return false;

    U64 epMask = 0ULL;
    if (is_white_move) {
        if (epFile>0) epMask |= (1ULL << (32+epFile-1));
        if (epFile<7) epMask |= (1ULL << (32+epFile+1));
        return (pieceBitboards[1] & colorBitboards[0] & epMask) != 0;
    } else {
        if (epFile>0) epMask |= (1ULL << (24+epFile-1));
        if (epFile<7) epMask |= (1ULL << (24+epFile+1));
        return (pieceBitboards[1] & colorBitboards[1] & epMask) != 0;
    }
}

void Board::updateFiftyMoveCounter(int moved_piece, bool isCapture) {
    if (isCapture || moved_piece == pawn) {
        currentGameState.fiftyMoveCounter = 0;
        //zobrist_history.clear();
        return;
    }
    
    currentGameState.fiftyMoveCounter++;

}

bool Board::isThreefold() {
    int count = 0;

    // Number of positions to look back: since last irreversible move
    int lookback = std::min<int>(currentGameState.fiftyMoveCounter + 1, zobrist_history.size());

    // Iterate backwards over just those positions
    for (int i = static_cast<int>(zobrist_history.size()) - lookback; i < static_cast<int>(zobrist_history.size()); ++i) {
        if (zobrist_history[i] == zobrist_hash)
            count++;
        if (count >= 3)
            return true;
    }

    return false;
}



// ------------------------------------------------------------
// Check detection
// ------------------------------------------------------------
bool Board::inCheck(bool init) {
    int side = init ? 1-move_color : move_color;
    int king_side = 1 - side;

    U64 king_bb = colorBitboards[king_side] & pieceBitboards[king];
    int king_sq = sqidx(king_bb);

    U64 occ = colorBitboards[0] | colorBitboards[1];

    // Knights
    if ((colorBitboards[side] & pieceBitboards[knight]) & PrecomputedMoveData::blankKnightAttacks[king_sq])
        return true;

    // Pawns                                 use king-side to basically say "if this king was a pawn, could it capture an enemy pawn" as the check
    if ((colorBitboards[side] & pieceBitboards[pawn]) & PrecomputedMoveData::fullPawnAttacks[king_sq][king_side])
        return true;

    // Bishops / Queens
    U64 diag_sliders = (colorBitboards[side] & pieceBitboards[bishop]) | (colorBitboards[side] & pieceBitboards[queen]);
    if (diag_sliders & Magics::bishopAttacks(king_sq, occ)) return true;

    // Rooks / Queens
    U64 ortho_sliders = (colorBitboards[side] & pieceBitboards[rook]) | (colorBitboards[side] & pieceBitboards[queen]);
    if (ortho_sliders & Magics::rookAttacks(king_sq, occ)) return true;

    return false;
}

bool Board::givesCheck(Move move) {
    // due to discovered checks, promotions, etc
    // we cannot just use bitboards to detect the check
    // so just make/unmake the move (fast enoough, but could use some nps testing)

    MakeMove(move);
    bool in_check = inCheck(true);
    UnmakeMove(move);

    return in_check;
}

// ------------------------------------------------------------
// FEN handling
// ------------------------------------------------------------
std::string Board::getBoardFEN() { setBoardFEN(); return fen; }

void Board::setFromFEN(std::string _fen) {
    hash_history.clear();
    zobrist_history.clear();

    std::istringstream fenStream(_fen);
    std::string boardState, turn, castling_rights, ep;
    int fifty_move = 0, full_moves = 1, piece;

    fen = _fen;
    fenStream >> boardState >> turn >> castling_rights >> ep >> fifty_move >> full_moves;
 
    // Reset state
    currentGameState = GameState();
    allGameMoves.clear();
    for (int i=0;i<6;i++) pieceBitboards[i]=0ULL;
    for (int i=0;i<2;i++) colorBitboards[i]=0ULL;
    std::fill(std::begin(sqToPiece), std::end(sqToPiece), -1);

    int row=7, col=0;
    for (char square : boardState) {
        if (square=='/') { row--; col=0; }
        else if (isdigit(square)) col += square-'0';
        else {
            piece = piece_int(static_cast<char>(tolower(square)));
            pieceBitboards[piece] |= (1ULL << (row*8+col));
            if (islower(square)) { colorBitboards[1] |= (1ULL << (row*8+col)); putPiece(piece+6,row*8+col);}
            else { colorBitboards[0] |= (1ULL << (row*8+col)); putPiece(piece,row*8+col);}
            sqToPiece[row*8+col] = piece + (islower(square) ? 6 : 0);
            col++;
        }
    }

    is_white_move = (turn=="w");
    currentGameState.castlingRights = 0;
    if (castling_rights!="-") {
        if (castling_rights.find("K")!=std::string::npos) currentGameState.castlingRights |= 0x1;
        if (castling_rights.find("Q")!=std::string::npos) currentGameState.castlingRights |= 0x2;
        if (castling_rights.find("k")!=std::string::npos) currentGameState.castlingRights |= 0x4;
        if (castling_rights.find("q")!=std::string::npos) currentGameState.castlingRights |= 0x8;
    }
    currentGameState.enPassantFile = (ep=="-") ? -1 : static_cast<int>(file_char.find(ep.substr(0,1)));
    currentGameState.fiftyMoveCounter = fifty_move;
    plyCount = is_white_move ? (full_moves-1)*2 : (full_moves-1)*2+1;

    move_color = is_white_move ? 0 : 1;
    is_in_check = inCheck(true);
    gameStateHistory.push_back(currentGameState);

    initZobristKeys();
    zobrist_hash = computeZobristHash();
    hash_history[zobrist_hash]++;
    zobrist_history.push_back(zobrist_hash);
}

void Board::setBoardFEN() {
    fen = "";
    int emptyCount=0;
    char piece;

    for (int rank=7;rank>=0;rank--) {
        emptyCount=0;
        for (int file=0;file<8;file++) {
            int square_index=rank*8+file;
            piece = piece_label(getPieceAt(square_index,getSideAt(square_index)));
            if (piece!='.') {
                if (emptyCount>0) { fen += std::to_string(emptyCount); emptyCount=0; }
                fen += piece;
            } else emptyCount++;
        }
        if (emptyCount>0) fen += std::to_string(emptyCount);
        if (rank>0) fen += '/';
    }

    fen += ' ';
    fen += is_white_move ? 'w' : 'b';

    std::string castling_str="";
    if (currentGameState.HasKingsideCastleRight(true)) castling_str+="K";
    if (currentGameState.HasQueensideCastleRight(true)) castling_str+="Q";
    if (currentGameState.HasKingsideCastleRight(false)) castling_str+="k";
    if (currentGameState.HasQueensideCastleRight(false)) castling_str+="q";
    if (castling_str=="") castling_str="-";
    fen += ' ' + castling_str;

    if (currentGameState.enPassantFile==-1) fen += " -";
    else {
        fen += " "; // space before en passant
        fen += file_char[static_cast<size_t>(currentGameState.enPassantFile)];
        fen += (is_white_move ? '6' : '3'); // use char, not string
    }

    fen += " " + std::to_string(currentGameState.fiftyMoveCounter);
    fen += " " + std::to_string(plyCount/2+1);
}

// ------------------------------------------------------------
// print board
// ------------------------------------------------------------
void Board::print_board() {
    char pieces[64];
    for (int i=63;i>=0;i--) pieces[i]='.';

    for (int i=0;i<64;i++) {
        for (int piece=0;piece<6;piece++) {
            if (get_bit(pieceBitboards[piece],i)) {
                if (get_bit(colorBitboards[0],i)) {
                    switch(piece){case 0:pieces[i]='P';break;case 1:pieces[i]='N';break;case 2:pieces[i]='B';break;case 3:pieces[i]='R';break;case 4:pieces[i]='Q';break;case 5:pieces[i]='K';break;}
                } else {
                    switch(piece){case 0:pieces[i]='p';break;case 1:pieces[i]='n';break;case 2:pieces[i]='b';break;case 3:pieces[i]='r';break;case 4:pieces[i]='q';break;case 5:pieces[i]='k';break;}
                }
            }
        }
    }

    if (!allGameMoves.empty()) allGameMoves.back().PrintMove();
    std::cout << "\n\n";
    for (int row=7;row>=0;row--){
        std::cout << row+1 << "  ";
        for (int col=0;col<8;col++) std::cout << "[" << pieces[row*8+col] << "] ";
        std::cout << "\n";
    }
    std::cout << "\n    a   b   c   d   e   f   g   h\n\n";

    setBoardFEN();
    std::cout << fen << "\n";
    std::cout << "\nMove: " << plyCount/2 << "\n";
    std::cout << (is_white_move?"White to move":"Black to move") << "\n";
    std::cout << (is_in_check?"Check":"") << "\n";
    std::cout << "Castling Rights: " << currentGameState.castlingRights << "\n";
    std::cout << "50 Move Counter: " << currentGameState.fiftyMoveCounter << "\n";
    std::cout << "En Passant: " << currentGameState.enPassantFile << "\n";
    std::cout << "Captured Piece: " << currentGameState.capturedPieceType << "\n\n";
}

// ------------------------------------------------------------
// Zobrist hashing functions
// ------------------------------------------------------------

U64 Board::randomU64() {
    static std::mt19937_64 rng(42); // static - persistent
    return rng();
}

/*
void Board::initPolyglotKeys() {
    std::mt19937_64 rng(42); 
    
    for (int piece = 0; piece < 12; ++piece)
        for (int square = 0; square < 64; ++square)
            polyglot_table[piece][square] = Random64[piece*64 + square];

    for (int i = 0; i < 4; ++i)
        polyglot_castling[i] = Random64[768 + i];

    for (int i = 0; i < 8; ++i)
        polyglot_enpassant[i] = Random64[772 + i];

    polyglot_side_to_move = Random64[780];
}
*/

void Board::initZobristKeys() {
    std::mt19937_64 rng(42); 

    for (int piece = 0; piece < 12; ++piece)
        for (int square = 0; square < 64; ++square)
            zobrist_table[piece][square] = rng();

    for (int i = 0; i < 4; ++i)
        zobrist_castling[i] = rng();

    for (int i = 0; i < 8; ++i)
        zobrist_enpassant[i] = rng();

    zobrist_side_to_move = rng();
}

U64 Board::computeZobristHash() {
    U64 hash = 0;

    // Pieces
    for (int sq = 0; sq < 64; ++sq) {
        int piece = getMovedPiece(sq);
        int color = getSideAt(sq);
        if (piece != -1) {
            int pieceIndex = (color == 0 ? 0 : 6) + piece;
            hash ^= zobrist_table[pieceIndex][sq];
        }
    }

    // Side to move
    if (!is_white_move)
        hash ^= zobrist_side_to_move;

    // Castling rights
    if (currentGameState.HasKingsideCastleRight(true))  hash ^= zobrist_castling[0];
    if (currentGameState.HasQueensideCastleRight(true)) hash ^= zobrist_castling[1];
    if (currentGameState.HasKingsideCastleRight(false)) hash ^= zobrist_castling[2];
    if (currentGameState.HasQueensideCastleRight(false)) hash ^= zobrist_castling[3];

    // En passant
    int epFile = currentGameState.enPassantFile;
    if (epFile >= 0 && epFile < 8)
        hash ^= zobrist_enpassant[epFile];

    return hash;
}


U64 Board::zobristCastlingHash(int castling_rights) {
    U64 hash = 0;
    if (castling_rights & 1) hash ^= zobrist_castling[0]; // K
    if (castling_rights & 2) hash ^= zobrist_castling[1]; // Q
    if (castling_rights & 4) hash ^= zobrist_castling[2]; // k
    if (castling_rights & 8) hash ^= zobrist_castling[3]; // q
    return hash;
}

void Board::auditZobrist(const Board &other, const std::string &label) const {
    bool mismatch = false;

    if (zobrist_hash != other.zobrist_hash) {
        std::cerr << label << " Zobrist hash mismatch: "
                  << std::hex << zobrist_hash << " vs " << other.zobrist_hash << "\n";
        mismatch = true;
    }

    if (is_white_move != other.is_white_move) {
        std::cerr << label << " Side-to-move differs: " 
                  << is_white_move << " vs " << other.is_white_move << "\n";
        mismatch = true;
    }

    if (currentGameState.castlingRights != other.currentGameState.castlingRights) {
        std::cerr << label << " Castling rights differ: " 
                  << currentGameState.castlingRights << " vs " 
                  << other.currentGameState.castlingRights << "\n";
        mismatch = true;
    }

    if (currentGameState.enPassantFile != other.currentGameState.enPassantFile) {
        std::cerr << label << " En-passant file differs: " 
                  << currentGameState.enPassantFile << " vs " 
                  << other.currentGameState.enPassantFile << "\n";
        mismatch = true;
    }

    for (int sq = 0; sq < 64; ++sq) {
        int p1 = getMovedPiece(sq);
        int p2 = other.getMovedPiece(sq);
        if (p1 != p2) {
            std::cerr << label << " Piece mismatch at square " << sq 
                      << ": " << p1 << " vs " << p2 << "\n";
            mismatch = true;
        }
    }

    if (!mismatch) {
        std::cerr << label << " Board hashes and components match.\n";
    }
}


/*
U64 Board::computePolyglotHash() {
    U64 hash = 0;

    std::cout << "Computing Zobrist hash step by step:\n";

    // Pieces
    for (int sq = 0; sq < 64; ++sq) {
        int piece = getMovedPiece(sq);
        int color = getSideAt(sq);
        if (piece != -1) {
            U64 key = zobrist_table[polyglotPieceHash(piece, color)][sq];
            hash ^= key;
            std::cout << "- XOR piece " << piece_label(piece)
                      << " (" << (color==0?"White":"Black") << ") at square " << sq
                      << ": key=0x" << std::hex << key
                      << ", hash now=0x" << hash << std::dec << "\n";
        }
    }

    // Side to move
    if (is_white_move) {
        hash ^= zobrist_side_to_move;
        std::cout << "- XOR side to move (White): key=0x" << std::hex << zobrist_side_to_move
                  << ", hash now=0x" << hash << std::dec << "\n";
    }

    // Castling rights
    struct { bool right; int idx; const char* name; } castling[4] = {
        { currentGameState.HasKingsideCastleRight(true), 0, "White K" },
        { currentGameState.HasQueensideCastleRight(true), 1, "White Q" },
        { currentGameState.HasKingsideCastleRight(false), 2, "Black K" },
        { currentGameState.HasQueensideCastleRight(false), 3, "Black Q" }
    };
    for (auto &c : castling) {
        if (c.right) {
            hash ^= zobrist_castling[c.idx];
            std::cout << "- XOR castling " << c.name
                      << ": key=0x" << std::hex << zobrist_castling[c.idx]
                      << ", hash now=0x" << hash << std::dec << "\n";
        }
    }

    // En passant
    int epFile = currentGameState.enPassantFile;
    if (epFile >= 0 && isEpHashable(epFile, is_white_move)) {
        hash ^= polyglot_enpassant[epFile];
        std::cout << "- XOR en passant file " << epFile
                  << ": key=0x" << std::hex << zobrist_enpassant[epFile]
                  << ", hash now=0x" << hash << std::dec << "\n";
    }



    return hash;
}


// pieces are encoded via idx= (1-move_color) + 2*piece_type
    // piece_type: pawn = 0, knight = 1, bishop = 2, rook = 3, queen = 4, king = 5
    // piece encoded so b-p =0, w-p =1, b-n=2, w-n=3, etc..
U64 Board::polyglotPieceHash(int piece, int color) {
    return (piece * 2) + (color ^ 1);
}

U64 Board::polyglotCastlingHash(int castling_rights) {
    U64 hash = 0;
    if (castling_rights & 1) hash ^= polyglot_castling[0]; // K
    if (castling_rights & 2) hash ^= polyglot_castling[1]; // Q
    if (castling_rights & 4) hash ^= polyglot_castling[2]; // k
    if (castling_rights & 8) hash ^= polyglot_castling[3]; // q
    return hash;
}

bool Board::canEPCapture() {
    int epFile = currentGameState.enPassantFile;
    if (epFile < 0 || epFile > 7)
        return false;

    int epSquare = !is_white_move
                   ? epFile + 16  // rank 3
                   : epFile + 40; // rank 6

    if (epSquare < 0 || epSquare >= 64)
        return false;

    if (!is_white_move) {
        // black pawns are above ep_sq
        int sq1 = epSquare + 7;
        int sq2 = epSquare + 9;

        bool ok1 = (sq1 < 64 && getMovedPiece(sq1) == pawn && getSideAt(sq1) == black);
        bool ok2 = (sq2 < 64 && getMovedPiece(sq2) == pawn && getSideAt(sq2) == black);

        return ok1 || ok2;
    } else {
        // white pawns are below ep_sq
        int sq1 = epSquare - 9;
        int sq2 = epSquare - 7;

        bool ok1 = (sq1 >= 0 && getMovedPiece(sq1) == pawn && getSideAt(sq1) == white);
        bool ok2 = (sq2 >= 0 && getMovedPiece(sq2) == pawn && getSideAt(sq2) == white);

        return ok1 || ok2;
    }
}

bool Board::isEpHashable(int epFile, bool stmIsWhite) const {
    if (epFile < 0 || epFile > 7)
        return false;

    int epSquare = !stmIsWhite // if white just played then black to move so ep sq is r3
                   ? epFile + 2*8  // rank 3 
                   : epFile + 5*8; // rank 6 

    if (!stmIsWhite) {
        int sq1 = epSquare + 7;
        int sq2 = epSquare + 9;
        bool ok1 = (sq1 < 64 && getMovedPiece(sq1) == pawn && getSideAt(sq1) == black);
        bool ok2 = (sq2 < 64 && getMovedPiece(sq2) == pawn && getSideAt(sq2) == black);
        return ok1 || ok2;
    } else {
        int sq1 = epSquare - 9;
        int sq2 = epSquare - 7;
        bool ok1 = (sq1 >= 0 && getMovedPiece(sq1) == pawn && getSideAt(sq1) == white);
        bool ok2 = (sq2 >= 0 && getMovedPiece(sq2) == pawn && getSideAt(sq2) == white);
        return ok1 || ok2;
    }
}
*/


void Board::debugZobristDifference(uint64_t current_hash, uint64_t expected_hash) {
    uint64_t diff = current_hash ^ expected_hash;

    if (diff == 0) {
        std::cout << "Zobrist hashes are identical.\n";
        return;
    }

    std::cout << "\nZobrist mismatch detected!\n";
    std::cout << "Current hash: " << std::hex << current_hash 
              << ", Expected hash: " << expected_hash 
              << ", XOR diff: " << diff << std::dec << "\n";

    // Side to move
    if ((current_hash & zobrist_side_to_move) != (expected_hash & zobrist_side_to_move))
        std::cout << "- Side to move differs\n";

    // Castling rights
    for (int i = 0; i < 4; ++i)
        if ((current_hash & zobrist_castling[i]) != (expected_hash & zobrist_castling[i]))
            std::cout << "- Castling right " << i << " differs\n";

    // En passant
    for (int i = 0; i < 8; ++i)
        if ((current_hash & zobrist_enpassant[i]) != (expected_hash & zobrist_enpassant[i]))
            std::cout << "- En passant file " << i << " differs\n";

    // Pieces
    for (int piece = 0; piece < 6; ++piece) {
        for (int color = 0; color < 2; ++color) {
            for (int sq = 0; sq < 64; ++sq) {
                uint64_t key = zobrist_table[(move_color == 0 ? 0 : 6) + piece][sq];
                bool inCurrent = (current_hash & key) != 0;
                bool inExpected = (expected_hash & key) != 0;
                if (inCurrent != inExpected)
                    std::cout << "- " << (color==0?"White":"Black") << " "
                              << piece_label(piece) << " on square " << sq
                              << " differs\n";
            }
        }
    }
}



// ------------------------------------------------------------
// NNUE functions
// ------------------------------------------------------------
//void Board::setNNUE(NNUE* nnue_ptr) {
//    nnue = nnue_ptr;
//}