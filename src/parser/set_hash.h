#pragma once
struct pair_hash {
    std::size_t operator()(const std::pair<std::string, bool>& pair) const {
        auto hash_str = std::hash<std::string>{}(pair.first);
        std::size_t hash_bool = pair.second ? 1231 : 1237;
        return hash_str ^ (hash_bool * 0x9e3779b9 + (hash_str << 6) + (hash_str >> 2));
    }
};

struct set_hash {
    std::size_t operator()(const std::set<std::pair<std::string, int>>& set) const {
        std::size_t seed = 0;
        for (const auto& elem : set) {
            seed += pair_hash{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};