
#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <queue>

#include <unordered_set>
#include <unordered_map>

namespace bigo{
namespace ml {
  template<class T = float>
  class FeatureDB {
  private:
    uint32_t Dim;
    bool normalize;
    bool initialize;
    void *hnsw;
    void normalization(const T* src, T* dst, unsigned dim);
  public:
    FeatureDB(
      const std::string space_name,
      const unsigned dim,
      const unsigned M,
      const unsigned max_elements,
      const unsigned query_ef,
      const unsigned construction_ef,
      const std::string dtype = "float");
    ~FeatureDB();
    unsigned size();
    const std::vector<char> dump();
    bool load(const std::vector<char>& serial);
    bool insert(const std::vector<T>& data,const uint64_t label);
    std::vector<std::pair<float, uint64_t>> query(const std::vector<T>&data, unsigned k);
    std::vector<std::pair<float, uint64_t>> queryAndInsert(const std::vector<T>& data,const uint64_t label,const unsigned k);
    std::vector<std::pair<float, uint64_t>> queryAndInsert(const std::vector<T>& data,const uint64_t label,const float threshold,const unsigned k);
  };
}
}
