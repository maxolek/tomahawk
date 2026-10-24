#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <book.h>
#include <helpers.h>
#include <move.h>


bool PolyglotBook::load(const fs::path& path) {
    entries.clear();
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    while (file) {
        uint64_t key = read_u64_be(file);
        uint16_t move = read_u16_be(file);
        uint16_t weight = read_u16_be(file);
        uint32_t learn = read_u32_be(file);

        if (!file) break;

        std::string uciMove = polyglotMoveToUCI(move);
        entries[key].push_back({key, move, weight, learn});
    }

    //std::cout << "info string Loaded Polyglot book: " << entries.size() << " positions\n";
    return true;
}

std::vector<BookEntry> PolyglotBook::get_moves(uint64_t key) {
    std::vector<BookEntry> moves;
    auto it = entries.find(key);
    if (it != entries.end()) {
        moves = it->second; // copy all entries
    }
    return moves;
}

std::string PolyglotBook::pick_weighted_move(const std::vector<BookEntry> & moves) {
    int total = 0;
    for (auto &m : moves) total += m.weight;

    int r = rand() % total;
    for (auto &m : moves) {
        if (r < m.weight) return polyglotMoveToUCI(m.move);
        r -= m.weight;
    }
    return polyglotMoveToUCI(moves[0].move); // fallback - first option
}

std::string PolyglotBook::probeBestMove(uint64_t key) {
    #ifdef DEV
        ScopedTimer timer(T_BOOK_PROBE);
    #endif
    return pick_weighted_move(get_moves(key));
}

// Converts Polyglot 16-bit move to UCI
std::string PolyglotBook::polyglotMoveToUCI(uint16_t move) {
    // in our move-value we set from-to-flag
    // but polyglot uses to-from-flag
    int from = (move >> 6) & 0x3F;
    int to   = move & 0x3F;
    int prom = (move >> 12) & 0xF;

    std::string uci;
    uci += squareToString(from);
    uci += squareToString(to);

    if (prom) {
        static const char promMap[5] = {0, 'n', 'b', 'r', 'q'};
        uci += promMap[prom];
    }
    return uci;
}

std::string PolyglotBook::squareToString(int sq) {
    char file = static_cast<char>('a' + (sq % 8));
    char rank = static_cast<char>('1' + (sq / 8));
    return std::string() + file + rank;
}

void PolyglotBook::printBookSample(const std::string &path, int n) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open book: " << path << std::endl;
        return;
    }

    for (int i = 0; i < n && file; ++i) {
        uint64_t key_be;
        uint16_t move_be, weight_be;
        uint32_t learn_be;

        file.read(reinterpret_cast<char*>(&key_be), 8);
        file.read(reinterpret_cast<char*>(&move_be), 2);
        file.read(reinterpret_cast<char*>(&weight_be), 2);
        file.read(reinterpret_cast<char*>(&learn_be), 4);

        if (!file) break;

        // Show raw big-endian values
        std::cout << "Entry " << i << " raw:" << std::endl;
        std::cout << " key    = " << std::hex << key_be << std::endl;
        std::cout << " move   = " << std::hex << move_be << std::endl;
        std::cout << " weight = " << std::hex << weight_be << std::endl;
        std::cout << " learn  = " << std::hex << learn_be << std::endl;
        std::cout << "-----------------------" << std::endl;
    }
}
