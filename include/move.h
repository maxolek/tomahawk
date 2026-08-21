/*
16 bit move representation (to preserve memory during search)

ffffttttttssssss
bits 0-5: start square index
bits 6-11: target square index
bits 12-15: flag (promotion type, etc)
*/

#ifndef MOVE_H
#define MOVE_H

#include "helpers.h"

struct Move 
{
private:
    ushort moveValue;
public:
    // flags
    static constexpr int noFlag = 0b0000;
    static constexpr int enPassantCaptureFlag = 0b0001;
    static constexpr int castleFlag = 0b0010;
    static constexpr int pawnTwoUpFlag = 0b0011;

    static constexpr int promoteToQueenFlag = 0b0100;
    static constexpr int promoteToKnightFlag = 0b0101;
    static constexpr int promoteToRookFlag = 0b0110;
    static constexpr int promoteToBishopFlag = 0b0111;

    // masks
    static constexpr ushort startSquareMask = 0b0000000000111111;
    static constexpr ushort targetSquareMask = 0b0000111111000000;
    static constexpr ushort flagMask = 0b1111000000000000;

    // constructors
    Move () {moveValue = 0;}

    Move (ushort _moveValue) {
        moveValue = _moveValue;
    }

    Move (int _startSquare, int _targetSquare) {
        moveValue = (ushort)(_startSquare | _targetSquare << 6);
    }

    Move (int _startSquare, int _targetSquare, int _flag) {
        moveValue = (ushort)(_startSquare | _targetSquare << 6 | _flag << 12);
    }

    Move (std::string uci) { // ep handled by board (no known info here)
        std::string _start, _target;
        char _flag;
        int start, target;

        _start = uci.substr(0,2); _target = uci.substr(2,2);
        start = algebraic_to_square(_start); target = algebraic_to_square(_target);

        if (uci == "e1g1" || uci == "e1c1" || uci == "e8g8" || uci == "e8c8") {
            // castling
            moveValue = (ushort)(start | target << 6 | castleFlag << 12);
        } else if (uci.length() == 5 ) { // promotions
            _flag = uci[4];
            switch (_flag) {
                    case 'q': moveValue = (ushort)(start | target << 6 | promoteToQueenFlag << 12); break;
                    case 'r': moveValue = (ushort)(start | target << 6 | promoteToRookFlag<< 12); break;
                    case 'b': moveValue = (ushort)(start | target << 6 | promoteToBishopFlag<< 12); break;
                    case 'n': moveValue = (ushort)(start | target << 6 | promoteToKnightFlag<< 12); break;
                }
        } else {moveValue = (ushort)(start | target << 6);} // rest
        
    }

    // getters
    ushort Value() const { return moveValue; }
    bool IsNull() const { return moveValue == 0; }
    int StartSquare() const { return moveValue & startSquareMask; }
    int TargetSquare() const { return ( moveValue & targetSquareMask) >> 6; }
    bool IsPromotion() const { return MoveFlag() >= promoteToQueenFlag; }
    int MoveFlag() const { return moveValue >> 12; }

    void PrintMove() const { 
        int start = StartSquare();
        int target = TargetSquare();
        int promo_int = PromotionPieceType(); 
        char promo_char = (promo_int > 0) ? piece_label(promo_int) : '.';

        std::cout
            << square_to_algebraic(StartSquare())
            << "->"
            << square_to_algebraic(TargetSquare());
        if (PromotionPieceType() > 0) {
            std::cout << " == " << piece_label(PromotionPieceType());
        }
        std::cout << '\n';
    }
    std::string uci() const {
        int start = StartSquare(); int target = TargetSquare(); 
        if (IsPromotion())
            return square_to_algebraic(start) + square_to_algebraic(target) + piece_label(PromotionPieceType()+10); // +10 is lower case
        return square_to_algebraic(start) + square_to_algebraic(target);
    }

    int PromotionPieceType() const {
        switch (MoveFlag()) {
            case promoteToQueenFlag:
                return queen;
            case promoteToRookFlag:
                return rook;
            case promoteToBishopFlag:
                return bishop;
            case promoteToKnightFlag:
                return knight;
            default:
                return -1;
        }
    }

    // statics
    static Move NullMove() { return Move(0); }
    static bool SameMove(Move a, Move b) { return a.moveValue == b.moveValue; }

};

// generalized printing method
inline std::ostream& operator<<(std::ostream& os, const Move& m) {
    if (m.IsNull()) {
        os << "null";
    } else {
        os << m.uci();
    }
    return os;
}

#endif