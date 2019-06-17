
#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <queue>

#include <unordered_set>
#include <unordered_map>

namespace fdb {
  template<class T = float>
  class FeatureDB {
  private:
    uint32_t Dim;
    bool normalize;
    bool initialize;
    void *hnsw;
    void normalization(T* src, T* dst, unsigned dim);
  public:
    FeatureDB(
      std::string space_name,
      const unsigned dim,
      const unsigned M,
      const unsigned max_elements,
      const unsigned query_ef,
      const unsigned construction_ef,
      std::string dtype = "float");
    ~FeatureDB();
    unsigned size();
    const std::vector<char> dump();
    bool load(const std::vector<char>& serial);
    bool insert(std::vector<T> data, uint64_t label);
    std::vector<std::pair<float, uint64_t>> query(const std::vector<T>&data, unsigned k);
    std::vector<std::pair<float, uint64_t>> queryAndInsert(const std::vector<T>& data, uint64_t label, unsigned k);
    std::vector<std::pair<float, uint64_t>> queryAndInsert(const std::vector<T>& data, uint64_t label, float threshold, unsigned k);
  };
}
