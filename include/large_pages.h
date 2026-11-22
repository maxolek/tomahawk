#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace mem {

enum class PageMode {
    Standard,
    Large,
    TransparentHugePageHint
};

struct Allocation {
    void* ptr = nullptr;
    std::size_t allocated_bytes = 0;
    PageMode page_mode = PageMode::Standard;
};

inline std::size_t round_up(std::size_t value, std::size_t multiple) {
    if (multiple == 0) return value;
    std::size_t remainder = value % multiple;
    if (remainder == 0) return value;
    const std::size_t padding = multiple - remainder;
    if (value > static_cast<std::size_t>(-1) - padding) return 0;
    return value + padding;
}

#ifdef _WIN32

inline bool enable_lock_memory_privilege() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
        return false;
    }

    TOKEN_PRIVILEGES tp{};
    if (!LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
        CloseHandle(token);
        return false;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    BOOL ok = AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr);
    DWORD err = GetLastError();
    CloseHandle(token);

    // AdjustTokenPrivileges can return TRUE even when ERROR_NOT_ALL_ASSIGNED occurs
    return ok && (err == ERROR_SUCCESS);
}

inline Allocation alloc_large_pages(std::size_t requested_bytes) {
    Allocation result{};
    if (requested_bytes == 0) return result;

    static bool privilege_attempted = false;
    static bool privilege_enabled = false;
    if (!privilege_attempted) {
        privilege_enabled = enable_lock_memory_privilege();
        privilege_attempted = true;
    }

    if (privilege_enabled) {
        SIZE_T large_page_min = GetLargePageMinimum();
        if (large_page_min > 0) {
            std::size_t aligned_size = round_up(requested_bytes, static_cast<std::size_t>(large_page_min));
            void* p = VirtualAlloc(
                nullptr,
                aligned_size,
                MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                PAGE_READWRITE
            );
            if (p != nullptr) {
                result.ptr = p;
                result.allocated_bytes = aligned_size;
                result.page_mode = PageMode::Large;
                return result;
            }
        }
    }

    // Graceful fallback: standard VirtualAlloc (64 KB granularity, page-aligned)
    void* fallback = VirtualAlloc(nullptr, requested_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (fallback != nullptr) {
        result.ptr = fallback;
        result.allocated_bytes = requested_bytes;
        result.page_mode = PageMode::Standard;
    }
    return result;
}

inline void free_large_pages(Allocation& alloc) {
    if (alloc.ptr != nullptr) {
        VirtualFree(alloc.ptr, 0, MEM_RELEASE);
        alloc.ptr = nullptr;
        alloc.allocated_bytes = 0;
        alloc.page_mode = PageMode::Standard;
    }
}

#else

// Linux / POSIX fallback using Transparent Huge Pages (THP) hint
inline Allocation alloc_large_pages(std::size_t requested_bytes) {
    Allocation result{};
    if (requested_bytes == 0) return result;

    constexpr std::size_t huge_page_size = 2 * 1024 * 1024;
    std::size_t aligned_size = round_up(requested_bytes, huge_page_size);
    if (aligned_size == 0) return result;

    void* p = nullptr;
    if (posix_memalign(&p, huge_page_size, aligned_size) == 0 && p != nullptr) {
        PageMode page_mode = PageMode::Standard;
#ifdef MADV_HUGEPAGE
        if (madvise(p, aligned_size, MADV_HUGEPAGE) == 0)
            page_mode = PageMode::TransparentHugePageHint;
#endif
        result.ptr = p;
        result.allocated_bytes = aligned_size;
        result.page_mode = page_mode;
    }
    return result;
}

inline void free_large_pages(Allocation& alloc) {
    if (alloc.ptr != nullptr) {
        std::free(alloc.ptr);
        alloc.ptr = nullptr;
        alloc.allocated_bytes = 0;
        alloc.page_mode = PageMode::Standard;
    }
}

#endif

} // namespace mem
