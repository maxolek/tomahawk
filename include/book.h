#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include "search_stats.h"
#include "timer.h"

struct BookEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};  

class PolyglotBook {
private:
public:
    std::unordered_map<uint64_t, std::vector<BookEntry>> entries;

    bool load(const fs::path& path);

    std::vector<BookEntry> get_moves(uint64_t key);
    std::string pick_weighted_move(const std::vector<BookEntry> & moves);
    std::string probeBestMove(uint64_t key);

    // Converts Polyglot 16-bit move to UCI
    std::string polyglotMoveToUCI(uint16_t move);
    std::string squareToString(int sq);

    void printBookSample(const std::string &path, int n = 10);
};
