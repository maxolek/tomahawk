// affinity.cpp
// Search-thread CPU placement (see affinity.h).

#include "affinity.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <thread>

#if defined(_WIN32)
    #if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601 // processor groups and GetLogicalProcessorInformationEx
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#elif defined(__linux__)
    #include <fstream>
    #include <pthread.h>
    #include <sched.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

namespace affinity {
namespace {

constexpr int MAX_CPU = 4095; // largest CPU number accepted in an option value

// One physical core.
struct Core {
    std::vector<int> cpus; // its logical CPUs
    int speed = 0;         // core type: higher is faster (Windows efficiency class, Linux capacity)
    int cluster = -1;      // id of the L2 cache it msay share with other cores; -1 = unknown
};

// The cores `auto` treats as performance cores: within 10% of the fastest type.
bool is_fast(int speed, int fastest) { return speed * 10 >= fastest * 9; }

bool is_separator(char c) { return c == ',' || std::isspace(static_cast<unsigned char>(c)); }

// "0-3,8 10": CPU numbers and ranges separated by commas and/or white space.
bool parse_cpu_list(const std::string& text, std::vector<int>& out) {
    out.clear();
    size_t i = 0;
    auto number = [&](int& value) {
        if (i >= text.size() || !std::isdigit(static_cast<unsigned char>(text[i]))) return false;
        value = 0;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            value = value * 10 + (text[i++] - '0');
            if (value > MAX_CPU) return false;
        }
        return true;
    };
    while (i < text.size() && is_separator(text[i])) ++i;
    while (i < text.size()) {
        int first = 0, last = 0;
        if (!number(first)) return false;
        last = first;
        if (i < text.size() && text[i] == '-') {
            ++i;
            if (!number(last) || last < first) return false;
        }
        if (i < text.size() && !is_separator(text[i])) return false;
        for (int cpu = first; cpu <= last; ++cpu) out.push_back(cpu);
        while (i < text.size() && is_separator(text[i])) ++i;
    }
    return true;
}

std::vector<int> sorted(std::vector<int> cpus) {
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

Topology build_topology(std::vector<Core> cores) {
    Topology topo;
    cores.erase(std::remove_if(cores.begin(), cores.end(), [](const Core& c) { return c.cpus.empty(); }),
                cores.end());
    if (cores.empty()) return topo;
    for (Core& core : cores) core.cpus = sorted(core.cpus);
    std::sort(cores.begin(), cores.end(), [](const Core& a, const Core& b) {
        return a.speed != b.speed ? a.speed > b.speed : a.cpus[0] < b.cpus[0];
    });
    const int fastest = cores[0].speed;

    // Core order: fastest type first. Within a type, round-robin over the L2
    // clusters, so consecutive threads go to different efficiency-core clusters
    // (a core with a private L2 is a cluster of one and keeps its place).
    std::vector<const Core*> core_order;
    for (size_t begin = 0; begin < cores.size();) {
        size_t end = begin;
        while (end < cores.size() && cores[end].speed == cores[begin].speed) ++end;
        std::vector<int> ids;
        std::vector<std::vector<const Core*>> clusters;
        for (size_t k = begin; k < end; ++k) {
            const int id = cores[k].cluster;
            const auto it = std::find(ids.begin(), ids.end(), id);
            if (id < 0 || it == ids.end()) {
                ids.push_back(id);
                clusters.push_back({&cores[k]});
            } else {
                clusters[static_cast<size_t>(it - ids.begin())].push_back(&cores[k]);
            }
        }
        for (size_t round = 0, added = 1; added; ++round) {
            added = 0;
            for (const auto& cluster : clusters)
                if (round < cluster.size()) {
                    core_order.push_back(cluster[round]);
                    ++added;
                }
        }
        begin = end;
    }

    // One logical CPU of every core first, then the SMT siblings.
    size_t width = 0;
    for (const Core& core : cores) width = (std::max)(width, core.cpus.size());
    for (size_t t = 0; t < width; ++t)
        for (const Core* core : core_order)
            if (t < core->cpus.size()) topo.order.push_back(core->cpus[t]);

    for (const Core& core : cores) {
        topo.cores.push_back(core.cpus);
        if (is_fast(core.speed, fastest)) {
            ++topo.performance_cores;
            topo.performance.insert(topo.performance.end(), core.cpus.begin(), core.cpus.end());
        } else {
            ++topo.other_cores;
        }
        topo.smt = topo.smt || core.cpus.size() > 1;
    }
    topo.performance = sorted(topo.performance);
    topo.supported = true;
    return topo;
}

// ------------------------------------------------------------
// Platform layer: cores, CPUs the process may use, pinning
// ------------------------------------------------------------

#if defined(_WIN32)

constexpr int GROUP_BITS = static_cast<int>(sizeof(KAFFINITY) * 8);

std::vector<Core> detect_cores() {
    std::vector<Core> cores;
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &length);
    if (length == 0) return cores;
    std::vector<unsigned char> buffer(length);
    if (!GetLogicalProcessorInformationEx(
            RelationAll, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length))
        return cores;

    std::vector<GROUP_AFFINITY> l2; // the logical CPUs sharing each L2 cache
    for (DWORD offset = 0; offset < length;) {
        const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (info->Size == 0) break;
        if (info->Relationship == RelationProcessorCore) {
            Core core;
            core.speed = info->Processor.EfficiencyClass;
            const GROUP_AFFINITY& group = info->Processor.GroupMask[0]; // a core is in one group
            for (int bit = 0; bit < GROUP_BITS; ++bit)
                if ((group.Mask >> bit) & 1) core.cpus.push_back(64 * group.Group + bit);
            cores.push_back(core);
        } else if (info->Relationship == RelationCache && info->Cache.Level == 2) {
            l2.push_back(info->Cache.GroupMask);
        }
        offset += info->Size;
    }
    for (Core& core : cores) {
        if (core.cpus.empty()) continue;
        const int cpu = core.cpus[0];
        for (size_t k = 0; k < l2.size(); ++k)
            if (l2[k].Group == cpu / 64 && ((l2[k].Mask >> (cpu % 64)) & 1)) {
                core.cluster = static_cast<int>(k);
                break;
            }
    }
    return cores;
}

// The process affinity mask (inherited from the program that started the engine).
bool process_mask(DWORD_PTR& mask) {
    DWORD_PTR system = 0;
    return GetActiveProcessorGroupCount() == 1 &&
           GetProcessAffinityMask(GetCurrentProcess(), &mask, &system) && mask != 0;
}

// A kernel32 function that older Windows versions lack, or nullptr.
template <typename F>
F kernel32_function(const char* name) {
    static_assert(sizeof(F) == sizeof(FARPROC), "function pointer size");
    F function = nullptr;
    if (const HMODULE kernel = GetModuleHandleW(L"kernel32.dll"))
        if (const FARPROC address = GetProcAddress(kernel, name)) std::memcpy(&function, &address, sizeof function);
    return function;
}

// CPUs of the process default CPU set (Windows 10+), the other way a program can
// assign CPUs to the engine: fastchess -use-affinity uses it on Windows 11. It does
// not show in the affinity mask, and a thread's affinity mask overrides it, so it
// has to be read here. Empty if the process has no default CPU set.
std::vector<int> process_cpu_set() {
    using GetDefaultCpuSets = BOOL(WINAPI*)(HANDLE, PULONG, ULONG, PULONG);
    using GetCpuSetInformation = BOOL(WINAPI*)(PVOID, ULONG, PULONG, HANDLE, ULONG);
    static const auto get_ids = kernel32_function<GetDefaultCpuSets>("GetProcessDefaultCpuSets");
    static const auto get_info = kernel32_function<GetCpuSetInformation>("GetSystemCpuSetInformation");
    if (!get_ids || !get_info) return {};
    ULONG count = 0;
    if (get_ids(GetCurrentProcess(), nullptr, 0, &count) || count == 0) return {}; // no default CPU set
    std::vector<ULONG> ids(count);
    if (!get_ids(GetCurrentProcess(), ids.data(), count, &count)) return {};
    ids.resize(count);
    std::sort(ids.begin(), ids.end());

    ULONG length = 0;
    get_info(nullptr, 0, &length, nullptr, 0);
    std::vector<unsigned char> info(length);
    if (length == 0 || !get_info(info.data(), length, &length, nullptr, 0)) return {};
    // SYSTEM_CPU_SET_INFORMATION entries, read by offset (the struct needs recent headers):
    // DWORD Size, DWORD Type (0 = CPU set), DWORD Id, WORD Group, BYTE LogicalProcessorIndex
    std::vector<int> cpus;
    for (ULONG offset = 0; offset + 15 <= length;) {
        DWORD size = 0, type = 0, id = 0;
        WORD group = 0;
        std::memcpy(&size, &info[offset], sizeof size);
        std::memcpy(&type, &info[offset + 4], sizeof type);
        std::memcpy(&id, &info[offset + 8], sizeof id);
        std::memcpy(&group, &info[offset + 12], sizeof group);
        if (size == 0) break;
        if (type == 0 && std::binary_search(ids.begin(), ids.end(), id)) cpus.push_back(64 * group + info[offset + 14]);
        offset += size;
    }
    return sorted(cpus);
}

std::vector<int> allowed_cpus(const Topology& topo) {
    std::vector<int> cpus;
    DWORD_PTR mask = 0;
    if (process_mask(mask)) {
        for (int bit = 0; bit < GROUP_BITS; ++bit)
            if ((mask >> bit) & 1) cpus.push_back(bit);
    } else {
        cpus = sorted(topo.order); // several processor groups: no single mask describes the process
    }
    const std::vector<int> cpu_set = process_cpu_set();
    std::vector<int> both;
    std::set_intersection(cpus.begin(), cpus.end(), cpu_set.begin(), cpu_set.end(), std::back_inserter(both));
    return both.empty() ? cpus : both;
}

bool pin_self(const std::vector<int>& cpus) {
    GROUP_AFFINITY group{};
    group.Group = static_cast<WORD>(cpus[0] / 64);
    for (int cpu : cpus)
        if (cpu / 64 == group.Group && cpu % 64 < GROUP_BITS) group.Mask |= KAFFINITY(1) << (cpu % 64);
    return SetThreadGroupAffinity(GetCurrentThread(), &group, nullptr) != 0;
}

int current_cpu() {
    PROCESSOR_NUMBER processor{};
    GetCurrentProcessorNumberEx(&processor);
    return 64 * processor.Group + processor.Number;
}

#elif defined(__linux__)

std::string read_line(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    return line;
}

std::vector<Core> detect_cores() {
    const std::string root = "/sys/devices/system/cpu/";
    std::vector<int> online;
    if (!parse_cpu_list(read_line(root + "online"), online) || online.empty()) return {};
    std::vector<int> intel_p; // hybrid Intel CPUs list their performance cores here
    if (!parse_cpu_list(read_line("/sys/devices/cpu_core/cpus"), intel_p)) intel_p.clear();

    std::map<int, Core> cores; // keyed by the first logical CPU of the core
    for (int cpu : online) {
        const std::string dir = root + "cpu" + std::to_string(cpu) + "/";
        std::vector<int> siblings;
        if (!parse_cpu_list(read_line(dir + "topology/thread_siblings_list"), siblings) || siblings.empty())
            siblings = {cpu};
        Core& core = cores[siblings[0]];
        core.cpus.push_back(cpu);
        const std::string capacity = read_line(dir + "cpu_capacity"); // asymmetric (big.LITTLE) systems
        if (!capacity.empty())
            core.speed = std::atoi(capacity.c_str());
        else if (!intel_p.empty())
            core.speed = std::find(intel_p.begin(), intel_p.end(), cpu) != intel_p.end() ? 1 : 0;
        for (int index = 0; index < 8; ++index) {
            const std::string cache = dir + "cache/index" + std::to_string(index) + "/";
            const std::string level = read_line(cache + "level");
            if (level.empty()) break;
            std::vector<int> shared;
            if (level == "2" && parse_cpu_list(read_line(cache + "shared_cpu_list"), shared) && !shared.empty()) {
                core.cluster = shared[0];
                break;
            }
        }
    }
    std::vector<Core> result;
    for (auto& entry : cores) result.push_back(entry.second);
    return result;
}

std::vector<int> affinity_of(pid_t thread) { // 0 = the calling thread
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<int> cpus;
    if (sched_getaffinity(thread, sizeof(set), &set) == 0)
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
    return cpus;
}

// Recorded at startup, in case the initial thread's affinity cannot be read later.
const std::vector<int> startup_cpus = affinity_of(0);

// The CPUs of the process: the affinity of its initial thread (thread id = process
// id), which the engine never pins (main() only waits for the UCI thread) and
// which `taskset -p` and match runners set, possibly after startup.
std::vector<int> allowed_cpus(const Topology& topo) {
    std::vector<int> cpus = affinity_of(getpid());
    if (cpus.empty()) cpus = startup_cpus;
    return cpus.empty() ? sorted(topo.order) : cpus;
}

bool pin_self(const std::vector<int>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus)
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

int current_cpu() {
    return sched_getcpu();
}

#else // no thread placement API (e.g. macOS)

std::vector<Core> detect_cores() { return {}; }
std::vector<int> allowed_cpus(const Topology&) { return {}; }
bool pin_self(const std::vector<int>&) { return false; }
int current_cpu() { return -1; }

#endif

// Where each of `count` threads may run when not pinned: every CPU in `allowed`.
// A Windows thread runs in one processor group only, so with several groups each
// thread gets a whole group, filling the groups in proportion to their size.
std::vector<std::vector<int>> unpinned_sets(const std::vector<int>& allowed, size_t count) {
#if defined(_WIN32)
    std::map<int, std::vector<int>> groups;
    for (int cpu : allowed) groups[cpu / 64].push_back(cpu);
    if (groups.size() > 1) {
        std::map<int, long> free_cpus;
        for (const auto& group : groups) free_cpus[group.first] = static_cast<long>(group.second.size());
        std::vector<std::vector<int>> sets;
        for (size_t i = 0; i < count; ++i) {
            auto best = free_cpus.begin();
            for (auto it = free_cpus.begin(); it != free_cpus.end(); ++it)
                if (it->second > best->second) best = it;
            --best->second;
            sets.push_back(groups[best->first]);
        }
        return sets;
    }
#endif
    return std::vector<std::vector<int>>(count, allowed);
}

} // namespace

// ------------------------------------------------------------
// Public interface
// ------------------------------------------------------------

const Topology& topology() {
    static const Topology topo = build_topology(detect_cores());
    return topo;
}

std::string describe(const Topology& topo) {
    if (!topo.supported) return "cpu placement is not supported on this platform";
    std::string text = std::to_string(topo.order.size()) + " logical cpus: ";
    if (topo.other_cores)
        text += std::to_string(topo.performance_cores) + " performance + " + std::to_string(topo.other_cores) +
                " efficiency cores";
    else
        text += std::to_string(topo.performance_cores) + " cores";
    if (topo.smt) text += " (smt)";
    return text + "; auto order " + format_cpus(topo.order);
}

std::string describe_current_core() {
    const int cpu = current_cpu();
    if (cpu < 0) return "current core unavailable";

    const Topology& topo = topology();
    for (const auto& core : topo.cores) {
        if (std::find(core.begin(), core.end(), cpu) != core.end()) {
            return "core cpus " + format_cpus(core) + " (running on logical cpu " +
                   std::to_string(cpu) + ")";
        }
    }
    return "logical cpu " + std::to_string(cpu) + " (core topology unavailable)";
}

bool parse(const std::string& text, Spec& spec, std::string& error) {
    std::string value;
    for (char c : text) value += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const size_t first = value.find_first_not_of(" \t\r\n");
    value = first == std::string::npos ? "" : value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);

    if (value.empty() || value == "auto" || value == "<empty>") { // UCI writes an empty string as <empty>
        spec = Spec{};
        return true;
    }
    if (value == "none" || value == "off") {
        spec = Spec{Spec::Mode::None, {}};
        return true;
    }
    std::vector<int> cpus;
    if (!parse_cpu_list(value, cpus) || cpus.empty()) {
        error = "expected auto, none, or a list of logical cpus such as 0,2,4-7";
        return false;
    }
    const Topology& topo = topology();
    if (!topo.supported) {
        error = "cpu lists are not supported on this platform";
        return false;
    }
    for (int cpu : cpus)
        if (std::find(topo.order.begin(), topo.order.end(), cpu) == topo.order.end()) {
            error = "there is no cpu " + std::to_string(cpu) + " (logical cpus: " + format_cpus(sorted(topo.order)) + ")";
            return false;
        }
    std::vector<int> ascending = cpus;
    std::sort(ascending.begin(), ascending.end());
    const auto repeated = std::adjacent_find(ascending.begin(), ascending.end());
    if (repeated != ascending.end()) { // two threads on one CPU would share it
        error = "cpu " + std::to_string(*repeated) + " is listed more than once";
        return false;
    }
    spec = Spec{Spec::Mode::List, cpus};
    return true;
}

std::string to_string(const Spec& spec) {
    switch (spec.mode) {
        case Spec::Mode::None: return "none";
        case Spec::Mode::List: return format_cpus(spec.cpus);
        default: return "auto";
    }
}

Plan make_plan(const Spec& spec, int thread_count) {
    Plan plan;
    const size_t count = static_cast<size_t>((std::max)(1, thread_count));
    const std::string name = "affinity " + to_string(spec) + ": ";
    const Topology& topo = topology();
    const std::vector<int> allowed = allowed_cpus(topo); // empty where placement is unsupported
    plan.unpinned = unpinned_sets(allowed, count);
    plan.threads = plan.unpinned;
    if (!topo.supported) {
        plan.summary = name + (spec.mode == Spec::Mode::None ? "" : "not supported on this platform, ") +
                       "search threads are not pinned";
        return plan;
    }
    if (spec.mode == Spec::Mode::None) {
        plan.summary = name + "search threads are not pinned";
        return plan;
    }
    auto usable = [&](int cpu) { return std::binary_search(allowed.begin(), allowed.end(), cpu); };

    if (spec.mode == Spec::Mode::Auto && count == 1) {
        // Any performance core: concurrent single-threaded engines spread over them.
        std::vector<int> fast;
        for (int cpu : topo.performance)
            if (usable(cpu)) fast.push_back(cpu);
        if (fast.empty()) {
            plan.summary = name + "no performance core available to the process, search thread is not pinned";
            return plan;
        }
        plan.threads[0] = fast;
        plan.summary = name + "search thread on cpus " + format_cpus(fast);
        return plan;
    }

    std::vector<int> order, unavailable;
    for (int cpu : spec.mode == Spec::Mode::List ? spec.cpus : topo.order)
        (usable(cpu) ? order : unavailable).push_back(cpu);
    if (order.empty()) {
        plan.summary = name + "none of these cpus is available to the process, search threads are not pinned";
        return plan;
    }
    // The engine keeps thread_count <= cpu_limit(), so each thread gets a CPU of its
    // own; the modulo only guards against the process affinity shrinking meanwhile.
    std::vector<int> assigned;
    for (size_t i = 0; i < count; ++i) {
        assigned.push_back(order[i % order.size()]);
        plan.threads[i] = {assigned.back()};
    }
    plan.summary = name + (assigned.size() > 1 ? "search threads on cpus " : "search thread on cpu ") +
                   format_cpus(assigned);
    if (spec.mode == Spec::Mode::List && !unavailable.empty())
        plan.summary += " (not available to the process: " + format_cpus(unavailable) + ")";
    return plan;
}

bool apply_to_current_thread(const std::vector<int>& cpus) {
    return cpus.empty() || pin_self(cpus);
}

int cpu_limit(const Spec& spec) {
    const Topology& topo = topology();
    if (!topo.supported) { // no placement: the logical CPUs the OS reports, if it does
        const unsigned count = std::thread::hardware_concurrency();
        return count > 0 ? static_cast<int>(count) : std::numeric_limits<int>::max();
    }
    const std::vector<int> allowed = allowed_cpus(topo);
    auto count_usable = [&](const std::vector<int>& cpus) {
        int usable = 0;
        for (int cpu : cpus)
            if (std::binary_search(allowed.begin(), allowed.end(), cpu)) ++usable;
        return usable;
    };
    // the same CPUs make_plan() pins to; if there are none, threads run unpinned on `allowed`
    const int pinnable = spec.mode == Spec::Mode::List ? count_usable(spec.cpus)
                       : spec.mode == Spec::Mode::Auto ? count_usable(topo.order)
                                                       : 0;
    return pinnable > 0 ? pinnable : (std::max)(1, static_cast<int>(allowed.size()));
}

std::string format_cpus(const std::vector<int>& cpus) {
    std::string text;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) ++j;
        if (!text.empty()) text += ',';
        text += std::to_string(cpus[i]);
        if (j >= i + 2) {
            text += "-" + std::to_string(cpus[j]);
            i = j + 1;
        } else {
            ++i;
        }
    }
    return text;
}

} // namespace affinity
