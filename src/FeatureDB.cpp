
#include "FeatureDB.h"
#include "HierarchicalNSW/hnsw.h"

namespace bigo{
namespace ml {

  typedef hnsw::HierarchicalNSW<half,half,float> INDEX;
  template<class T>
  FeatureDB<T>::FeatureDB(
    const std::string space_name,
    const unsigned dim,
    const unsigned M,
    const unsigned max_elements,
    const unsigned query_ef,
    const unsigned construction_ef,
    const std::string dtype)
    :Dim(dim) {
    normalize = false;
    initialize = false;
    gemv::DisT dist;
    if (space_name == "ip") dist = gemv::DisT::IP;
    else if (space_name == "l2") dist = gemv::DisT::L2;
    else if (space_name == "cosine") {
      dist = gemv::DisT::IP;
      normalize = true;
    }
    if(std::is_same<T,float>::value)
      hnsw = (void*)new INDEX(max_elements, M, dim, query_ef, construction_ef, dist);
    else
      printf("Only float is supported!");
  }

  template<class T>
  FeatureDB<T>::~FeatureDB() {
    delete (INDEX*)hnsw;
  }

  template<class T>
  void FeatureDB<T>::normalization(const T* src, T* dst,const unsigned dim) {}

  template<class T>
  unsigned FeatureDB<T>::size() {
    return ((INDEX*)hnsw)->size();
  }

  template<class T>
  const std::vector<char> FeatureDB<T>::dump() {
    return ((INDEX*)hnsw)->dump();
  }

  template<class T>
  bool FeatureDB<T>::load(const std::vector<char>& serial) {
    if (initialize) return false;
    initialize = true;
    ((INDEX*)hnsw)->load(serial);
    return true;
  }

#define CHECK_EQ(obj, src, rt) \
    if (obj != src) return rt;

#define __ToHalf__                                          \
  void* tmp = aligned_alloc(64,sizeof(float)*Dim);          \
  memcpy(tmp,&data[0],sizeof(float)*Dim);                   \
  gemv::KParams<float,half> kp;                             \
  kp.dim = Dim;                                             \
  kp.src.push_back((float*)tmp);                            \
  kp.dst.push_back((half*)tmp);                             \
  if(normalize)                                             \
    L2NormAVX(kp);                                          \
  else                                                      \
    Float2HalfAVX(kp);

  template<class T>
  bool FeatureDB<T>::insert(const std::vector<T>& data,const uint64_t label) {
    initialize = true;
    CHECK_EQ(data.size(), Dim, false)

    __ToHalf__
    ((INDEX*)hnsw)->insertPoint((half*)tmp, label);
    free(tmp);
    return true;
  }

  template<class T>
  std::vector<std::pair<float, uint64_t>> FeatureDB<T>::query(const std::vector<T>& data,const unsigned k) {
    std::vector<std::pair<float, uint64_t>> result;
    CHECK_EQ(data.size(), Dim, result)

    __ToHalf__
    auto top_candidate = ((INDEX*)hnsw)->searchKnn((half*)tmp, k);
    result.resize(top_candidate.size());
    unsigned count = top_candidate.size();
    for (int i = count - 1; i >= 0; --i) {
      result[i] = top_candidate.top();
      top_candidate.pop();
    }
    free(tmp);
    return result;
  }

  template<class T>
  std::vector<std::pair<float, uint64_t>> FeatureDB<T>::queryAndInsert(const std::vector<T>& data,const uint64_t label,const unsigned k) {
    std::vector<std::pair<float, uint64_t>> result;
    CHECK_EQ(data.size(), Dim, result)

    __ToHalf__
    auto top_candidate = ((INDEX*)hnsw)->searchKnn((half*)tmp, k);
    result.resize(top_candidate.size());
    unsigned count = top_candidate.size();
    for (int i = count - 1; i >= 0; --i) {
      result[i] = top_candidate.top();
      top_candidate.pop();
    }
    ((INDEX*)hnsw)->insertPoint((half*)tmp, label);
    free(tmp);
    return result;
  }

  template<class T>
  std::vector<std::pair<float, uint64_t>> FeatureDB<T>::queryAndInsert(const std::vector<T>& data,const uint64_t label,const float threshold,const unsigned k) {
    std::vector<std::pair<float, uint64_t>> result;
    CHECK_EQ(data.size(), Dim, result)

    __ToHalf__
    auto top_candidate = ((INDEX*)hnsw)->searchKnn((half*)tmp, k);
    result.resize(top_candidate.size());
    unsigned count = top_candidate.size();
    for (int i = count - 1; i >= 0; --i) {
      result[i] = top_candidate.top();
      top_candidate.pop();
    }
    if (result[0].first > threshold)
      ((INDEX*)hnsw)->insertPoint((half*)tmp, label);
    free(tmp);
    return result;
  }

  template class FeatureDB<float>;
}
}
