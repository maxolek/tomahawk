#include <helpers.h>

std::string file_char = "abcdefgh";
std::string results_string[] = {
    "NotStarted",
    "InProgress",
    "WhiteIsMated",
    "BlackIsMated",
    "Stalemate",
    "Repetition",
    "FiftyMoveRule",
    "InsufficientMaterial",
    "DrawByArbiter",
    "WhiteTimeout",
    "BlackTimeout",
    "WhiteIllegalMove",
    "BlackIllegalMove"
};


// print bitboard
void print_bitboard(U64 bitboard) {
    // loop over board ranks
    for (int rank = 7; rank >= 0; rank--) {
        // loop over board files
        for (int file = 0; file < 8; file++) {
            // init square
            int square = rank * 8 + file;

            // print ranks
            if (!file)
                std::cout << rank + 1 << "\t";

            // print bit state 
            std::cout << " " << ((get_bit(bitboard, square)) ? 1 : 0);
        }
        // print new line every rank
        std::cout << "\n";
    }

    // print board files
    std::cout << "\n\t a b c d e f g h";

    // print bitboard as unsigned decimal number
    std::cout << "\n\nBitboard:\t" << bitboard << std::endl;
}

// this engines encoding is in little endian
// but in case we need to read big endian we create these readers
uint64_t read_u64_be(std::ifstream &file) {
    uint8_t b[8];
    file.read(reinterpret_cast<char*>(b), 8);
    return
        (uint64_t(b[0]) << 56) |
        (uint64_t(b[1]) << 48) |
        (uint64_t(b[2]) << 40) |
        (uint64_t(b[3]) << 32) |
        (uint64_t(b[4]) << 24) |
        (uint64_t(b[5]) << 16) |
        (uint64_t(b[6]) << 8)  |
        uint64_t(b[7]);
}

uint16_t read_u16_be(std::ifstream &file) {
    uint8_t b[2];
    file.read(reinterpret_cast<char*>(b), 2);
    return (uint16_t(b[0]) << 8) | uint16_t(b[1]);
}

uint32_t read_u32_be(std::ifstream &file) {
    uint8_t b[4];
    file.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) |
           (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8)  |
           uint32_t(b[3]);
}

//                 N,   NE,    E,    SE,    S,     SW,     W,     NW
// (rank,file) = (1,0) (1,1) (0,1) (-1,1) (-1,0) (-1,-1) (0,-1) (1,-1)
int direction_index(int start_square, int target_square) {
   std::pair<int,int> map = direction_map(start_square,target_square);
   int rank_dir = map.first; int file_dir = map.second;
    
    switch (rank_dir) {
        case -1:
            switch (file_dir) {
                case -1: 
                    return 5;
                    break;
                case 0: 
                    return 4;
                    break;
                case 1: 
                    return 3;
                    break;
            }
            return -1;
        case 0:
            switch (file_dir) {
                case -1: 
                    return 6;
                    break;
                case 0: 
                    return -1; // start=target
                    break;
                case 1: 
                    return 2;
                    break;
            }
            return -1;
        case 1:
            switch (file_dir) {
                case -1: 
                    return 7;
                    break;
                case 0: 
                    return 0;
                    break;
                case 1: 
                    return 1;
                    break;
                break;
            }
            return -1;
        default:
            return -1;
            break;
    }
}

// rank, file direction map
std::pair<int,int> direction_map(int start_square, int target_square) {
    int a_file = start_square % 8;
    int b_file = target_square % 8;
    int a_rank = start_square / 8;
    int b_rank = target_square / 8;
    if (abs(b_file - a_file) == abs(b_rank - a_rank) || b_file == a_file || b_rank == a_rank) {
        int rank_dir = (target_square/8 - start_square/8 > 0) ? 1 : (target_square/8 - start_square/8 < 0) ? -1 : 0;
        int file_dir = (target_square%8 - start_square%8 > 0) ? 1 : (target_square%8 - start_square%8 < 0) ? -1 : 0;
        return std::make_pair(rank_dir,file_dir);
    } else {return std::make_pair(0,0);}
}

U64 isolateLSB(U64 x) {
    return x & -x;
}

U64 isolateMSB(U64 x) {
    return 1ULL << getMSB(x);
}

int countBits(U64 x) {
    return __builtin_popcountll(x);
}

int getMSB(U64 x) {
    if (x == 0) {return -1;}
    return 63 - __builtin_clzll(x);
}

int getLSB(U64 x) {
    return __builtin_ffsll(static_cast<long long>(x)) - 1;
}

// Function to get the square index from a bitboard with only one bit set
int sqidx(U64 bitboard) {
    // Assuming there is only one bit set
    if (!bitboard) return -1;
    return __builtin_ctzll(bitboard);  // __builtin_ctzll is for GCC/Clang
}

// Flip square for black piece evaluation
int mirror(int square) {
    return square ^ 56;
}

// Returns a mask of all bits below the given square (lower indices)
U64 bitsBelow(int sq) {
    if (sq == 0) return 0ULL;
    return (1ULL << sq) - 1;
}

// Returns a mask of all bits above the given square (higher indices)
U64 bitsAbove(int sq) {
    if (sq == 63) return 0ULL;
    return ~((1ULL << (sq + 1)) - 1);
}

char piece_label(int piece) {
    switch (piece) {
        case -1:
            return '.';
            break;
        case 0:
            return 'P';
            break;
        case 1:
            return 'N';
            break;
        case 2:
            return 'B';
            break;
        case 3:
            return 'R';
            break;
        case 4:
            return 'Q';
            break;
        case 5:
            return 'K';
            break;
        case 10:
            return 'p';
            break;
        case 11:
            return 'n';
            break;
        case 12:
            return 'b';
            break;
        case 13:
            return 'r';
            break;
        case 14:
            return 'q';
            break;
        case 15:
            return 'k';
            break;
    }

    return '/';
}

int piece_int(char piece) {
    switch (piece) {
        case 'p':
            return 0;
        case 'n':
            return 1;
        case 'b':
            return 2;
        case 'r':
            return 3;
        case 'q':
            return 4;
        case 'k':
            return 5;
    }
    return -1;
}

// Converts a square index (0-63) to algebraic notation (a1-h8)
std::string square_to_algebraic(int square) {
    char file = static_cast<char>('a' + (square % 8)); // Get file (column)
    char rank = static_cast<char>('1' + (square / 8)); // Get rank (row)
    return std::string(1, file) + rank; // Combine file and rank
}

// convert an algebraic move (a-h,0-7) to square index (0-63)
int algebraic_to_square(std::string square) {
    char file = square[0];
    char rank = square[1];
    // Convert file to 0-based index (0 for 'a', 1 for 'b', ..., 7 for 'h')
    int file_index = file - 'a';
    // Convert rank to 0-based index (0 for '1', 1 for '2', ..., 7 for '8')
    int rank_index = rank - '1';
    // Calculate the index for the 1D array representing the board
    int index = (rank_index * 8) + file_index;
    return index;
}

// config functions
std::unordered_map<std::string, std::string> parse_kv(const fs::path& path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream f(path);
    if (!f) return kv;  // missing file = use defaults, not an error
    std::string line;
    while (std::getline(f, line)) {
        // strip comments and whitespace
        if (auto pos = line.find('#'); pos != std::string::npos) line.erase(pos);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        // trim
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };
        trim(k); trim(v);
        if (!k.empty()) kv[k] = v;
    }
    return kv;
}
